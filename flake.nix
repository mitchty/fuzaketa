{
  description = "fuzaketa flake";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    crane.url = "github:ipetkov/crane";

    flake-utils.url = "github:numtide/flake-utils";

    fenix.url = "github:nix-community/fenix";
    treefmt-nix.url = "github:numtide/treefmt-nix";

    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };

    mitchty.url = "github:mitchty/nix";
  };

  outputs =
    { self, ... }@inputs:
    inputs.flake-utils.lib.eachDefaultSystem (
      system:
      let
        metaCommon = desc: {
          description = if desc == "" then "fuzaketa" else "fuzaketa " + desc;
          mainProgram = "fuzaketa";
        };

        stableRust = (
          inputs.fenix.packages.${system}.stable.withComponents [
            "cargo"
            "clippy"
            "llvm-tools"
            "rustc"
            "rust-src"
            "rustfmt"
            "rust-analyzer"
          ]
        );

        pkgs = import inputs.nixpkgs {
          inherit system;
          overlays = [
            inputs.fenix.overlays.default
            inputs.mitchty.overlays.cargo-unused-features
          ];
        };

        pkgsCuda =
          if pkgs.stdenv.isLinux then
            import inputs.nixpkgs {
              inherit system;
              overlays = [ inputs.fenix.overlays.default ];
              config = {
                allowUnfree = true;
                cudaSupport = true;
              };
            }
          else
            null;

        inherit (pkgs) lib;

        craneLib = (inputs.crane.mkLib pkgs).overrideToolchain (_: stableRust);

        # Crane lib for CUDA builds, Linux only obvs.
        craneLibCuda =
          if pkgs.stdenv.isLinux then
            (inputs.crane.mkLib pkgsCuda).overrideToolchain (
              p:
              p.fenix.combine [
                p.fenix.stable.rustc
                p.fenix.stable.cargo
                p.fenix.stable.rust-std
              ]
            )
          else
            null;

        version = self.rev or self.dirtyShortRev or "nix-flake-cant-get-git-commit-sha";

        # Constrained src fileset to ensure cargo deps aren't rebuilt on
        # every change to files that don't contribute to the dependency
        # chain. Only Cargo.lock/Cargo.toml/build.rs are needed here for
        # cargo dependency rebuild detection.
        srcDeps = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [
            ./Cargo.lock
            ./Cargo.toml
            (lib.fileset.fileFilter (file: file.name == "Cargo.toml") ./crates)
            (lib.fileset.fileFilter (file: file.name == "build.rs") ./crates)
          ];
        };

        src = lib.fileset.toSource {
          root = ./.;
          fileset = lib.fileset.unions [
            (lib.fileset.fileFilter (file: file.hasExt "rs") ./crates)
            (lib.fileset.fileFilter (file: file.hasExt "toml") ./crates)
            ./Cargo.toml
            ./Cargo.lock
          ];
        };

        fileSetForCrate =
          crate:
          lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./Cargo.toml
              ./Cargo.lock
              (craneLib.fileset.commonCargoSources crate)
              (lib.fileset.fileFilter (file: file.hasExt "rs") ./crates)
              (lib.fileset.fileFilter (file: file.hasExt "toml") ./crates)
            ];
          };

        # burn's wgpu backend needs a Vulkan (Linux) / Metal (macOS, via
        # apple-sdk) / DX12 (Windows) adapter at runtime.
        commonXinputs = with pkgs; [
          vulkan-loader
          libx11
          libxcursor
          libxi
          libxrandr
          libxkbcommon
          wayland
        ];

        commonArgs = {
          inherit src;
          strictDeps = true;

          nativeBuildInputs =
            with pkgs;
            [
              git
              pkg-config
            ]
            ++ lib.optionals pkgs.stdenv.isLinux [
              mold
              llvmPackages.lld
            ];

          buildInputs =
            with pkgs;
            [ ]
            ++ lib.optionals pkgs.stdenv.hostPlatform.isLinux commonXinputs
            ++ lib.optionals pkgs.stdenv.isDarwin [
              apple-sdk
              rustPlatform.bindgenHook
              llvmPackages.libclang
            ];

          LD_LIBRARY_PATH = lib.optionalString pkgs.stdenv.isLinux (lib.makeLibraryPath commonXinputs);
          # Use clang as the C linker, fenix gcc-ld seems to be breaking with
          # latest update for -fuse-ld.
          CC = lib.optionalString pkgs.stdenv.isLinux "${pkgs.llvmPackages.clang}/bin/clang";
        };

        # Macos env vars shared between the derivation environment and the devshell env
        commonEnvDarwin = {
          LIBCLANG_PATH = lib.optionalString pkgs.stdenv.isDarwin "${pkgs.llvmPackages.libclang.lib}/lib";
          # cc-rs calls clang++ directly, bypassing whatever is on PATH. Point
          # it at libcxxClang which is a nix-wrapped clang++ that already has
          # libc++ headers wired in via -isystem, needed for any crate that
          # compiles C++ from source.
          CXX = lib.optionalString pkgs.stdenv.isDarwin "${pkgs.llvmPackages.libcxxClang}/bin/clang++";
        };

        # Use mold for faster linking on Linux.
        linuxMoldFlags = lib.optionalString pkgs.stdenv.isLinux "-C link-arg=-fuse-ld=mold";

        # Use nixpkgs cudatoolkit for headers and compilation setup
        cudaMerged = if pkgs.stdenv.isLinux then pkgsCuda.cudaPackages.cudatoolkit else null;

        # Common args for the fuzaketa-cuda build, has all the cuda runtime
        # junk in its trunk so that cubecl/burn-cuda can dlopen() and build
        # at runtime.
        commonArgsCuda =
          if pkgs.stdenv.isLinux then
            {
              inherit src;
              strictDeps = true;

              nativeBuildInputs = with pkgsCuda; [
                git
                pkg-config
                # autoAddDriverRunpath patches ELF RPATH entries so the CUDA
                # libs are found at runtime even outside of NixOS.
                autoAddDriverRunpath
                mold
                llvmPackages.lld
              ];

              buildInputs =
                with pkgsCuda;
                [
                  cudaPackages.cuda_cudart
                  cudaPackages.cuda_nvcc
                  cudaPackages.cuda_nvrtc
                  cudaPackages.cuda_cccl
                  cudaPackages.libcublas
                ]
                ++ commonXinputs;

              CC = "${pkgs.llvmPackages.clang}/bin/clang";
              CUDA_PATH = "${cudaMerged}";
              LD_LIBRARY_PATH = lib.makeLibraryPath (
                commonXinputs
                ++ (with pkgsCuda; [
                  cudaPackages.cuda_cudart
                  cudaPackages.cuda_nvrtc
                  cudaPackages.libcublas
                ])
              );
              RUSTFLAGS = linuxMoldFlags;
            }
          else
            { };

        nixEnvArgs = {
          NIX_GIT_REV = version;
        };

        devArgs = {
          CARGO_PROFILE = "dev";
        };

        releaseArgs = {
          CARGO_PROFILE = "release";
          RUSTFLAGS = "-D warnings ${lib.optionalString pkgs.stdenv.isLinux linuxMoldFlags}";
        };

        cargoArtifacts = craneLib.buildDepsOnly (
          commonArgs
          // devArgs
          // {
            src = srcDeps;
          }
        );

        cargoArtifactsRelease = craneLib.buildDepsOnly (
          commonArgs
          // releaseArgs
          // {
            src = srcDeps;
          }
        );

        cargoArtifactsCuda =
          if pkgs.stdenv.isLinux then
            craneLibCuda.buildDepsOnly (
              commonArgsCuda
              // nixEnvArgs
              // releaseArgs
              // {
                src = srcDeps;
                cargoExtraArgs = "-p fuzaketa --features fuzaketa/cuda";
              }
            )
          else
            null;

        individualCrateArgs = commonArgs // {
          inherit cargoArtifacts;
          doCheck = false;
        };

        # Release build of the fuzaketa binary using burn's wgpu backend
        # for running anywhere.
        fuzaketa = craneLib.buildPackage (
          commonArgs
          // nixEnvArgs
          // releaseArgs
          // {
            pname = "fuzaketa";
            version = version;
            cargoArtifacts = cargoArtifactsRelease;
            cargoExtraArgs = "-p fuzaketa --bin fuzaketa";
            src = fileSetForCrate ./crates/fuzaketa;
            doCheck = false;
            meta = metaCommon "training CLI (wgpu)";
          }
        );

        # Release build of the fuzaketa binary using the burn CUDA backend
        # for faster training/inference on Nvidia GPUs. Linux only obvs.
        fuzaketa-cuda =
          if pkgs.stdenv.isLinux then
            craneLibCuda.buildPackage (
              commonArgsCuda
              // nixEnvArgs
              // releaseArgs
              // {
                pname = "fuzaketa-cuda";
                version = version;
                cargoArtifacts = cargoArtifactsCuda;
                cargoExtraArgs = "-p fuzaketa --bin fuzaketa --features fuzaketa/cuda";
                src = fileSetForCrate ./crates/fuzaketa;
                doCheck = false;
                meta = metaCommon "training CLI (CUDA)" // {
                  platforms = [
                    "x86_64-linux"
                    "aarch64-linux"
                  ];
                };
              }
            )
          else
            null;

        treefmtEval = inputs.treefmt-nix.lib.evalModule pkgs {
          projectRootFile = "flake.nix";
          programs = {
            nixfmt.enable = true;
            rustfmt = {
              enable = true;
              edition = "2024";
            };
            taplo.enable = true;
          };
        };

        # These tools are made available in the git-hooks derivation PATH and
        # the devShell alike.
        hookTools = with pkgs; {
          inherit
            taplo
            nixfmt
            rustfmt
            git
            nix
            treefmt
            convco
            ;
        };

        # Instead of running nix flake check on each commit, just check at
        # push time. History can be rewritten to fix things up if it fails.
        git-hooks-check = inputs.git-hooks.lib.${system}.run {
          src = ./.;
          tools = hookTools;
          hooks = {
            nix-flake-check = {
              enable = true;
              name = "nix-flake-check";
              entry = "${pkgs.nix}/bin/nix flake check -L";
              language = "system";
              pass_filenames = false;
              stages = [ "pre-push" ];
              verbose = true;
            };
            commit-msg = {
              enable = true;
              name = "convco";
              entry = "${pkgs.lib.getExe pkgs.bash} -c '${pkgs.convco}/bin/convco check --from-stdin < \"$1\"' --";
              language = "system";
              stages = [ "commit-msg" ];
            };
            # Formatting is enforced by the separate `formatter` check below.
            treefmt.enable = false;
          };
        };
      in
      {
        checks = {
          formatter = treefmtEval.config.build.check self;
          git-hooks = git-hooks-check;

          fuzaketa-clippy = craneLib.cargoClippy (
            commonArgs
            // nixEnvArgs
            // devArgs
            // {
              inherit cargoArtifacts;
              cargoClippyExtraArgs = "--all-targets -- --deny warnings";
            }
          );
        };

        packages = {
          inherit fuzaketa;
          default = fuzaketa;
          clippy = self.checks.${system}.fuzaketa-clippy;
        }
        // lib.optionalAttrs pkgs.stdenv.isLinux {
          inherit fuzaketa-cuda;
        };

        apps = {
          fuzaketa = {
            type = "app";
            program = "${fuzaketa}/bin/fuzaketa";
            meta = metaCommon "training CLI (wgpu)";
          };
          default = self.apps.${system}.fuzaketa;
        }
        // lib.optionalAttrs pkgs.stdenv.isLinux {
          fuzaketa-cuda = {
            type = "app";
            program = "${fuzaketa-cuda}/bin/fuzaketa";
            meta = metaCommon "training CLI (CUDA)";
          };
        };

        formatter = treefmtEval.config.build.wrapper;

        devShells.default = craneLib.devShell (
          {
            checks = self.checks.${system};

            packages =
              (with pkgs; [
                cargo-edit
                cargo-outdated
                cargo-unused-features
                stableRust
              ])
              ++ (lib.attrValues hookTools)
              ++ commonArgs.buildInputs
              ++ commonArgs.nativeBuildInputs
              # CUDA 12 dev tools for linux devshell abusage
              ++ lib.optionals pkgs.stdenv.isLinux (
                with pkgsCuda;
                [
                  cudaPackages.cuda_cudart
                  cudaPackages.cuda_nvcc
                  cudaPackages.cuda_nvrtc
                  cudaPackages.cuda_cccl
                  cudaPackages.libcublas
                  autoAddDriverRunpath
                  # Mold must be on PATH for -fuse-ld=mold to work.
                  pkgs.mold
                ]
              );

            shellHook = ''
              ${git-hooks-check.shellHook}
            '';

            # Make sure eglot+etc.. pick the right rust-src for eglot/lsp mode
            # stuff using direnv.
            RUST_SRC_PATH = "${stableRust}/lib/rustlib/src/rust/library";

            # Use mold for faster linking in devshell interactive builds.
            RUSTFLAGS = lib.optionalString pkgs.stdenv.isLinux linuxMoldFlags;

            LD_LIBRARY_PATH =
              commonArgs.LD_LIBRARY_PATH
              + lib.optionalString pkgs.stdenv.isLinux (
                ":"
                + lib.makeLibraryPath (
                  with pkgsCuda;
                  [
                    cudaPackages.cuda_cudart
                    cudaPackages.cuda_nvrtc
                    cudaPackages.libcublas
                  ]
                )
              );

            # CUDA toolkit path used to find headers at runtime for jit compilation
            CUDA_PATH = lib.optionalString pkgs.stdenv.isLinux "${cudaMerged}";
          }
          // lib.optionalAttrs pkgs.stdenv.isDarwin commonEnvDarwin
        );
      }
    );
}
