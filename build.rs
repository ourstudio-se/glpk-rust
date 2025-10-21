use std::path::PathBuf;
use std::process::Command;
use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // Strategy 1: Try system GLPK (pkg-config + direct detection)
    if try_system_glpk() {
        println!("cargo:warning=Using system GLPK");
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        generate_bindings_simple(&out_dir);
        return;
    }
    
    // Strategy 2: Try automatic installation then detect again
    println!("cargo:warning=GLPK not found, attempting automatic installation...");
    if try_install_system_glpk() {
        // After installation, try detection again
        if try_system_glpk() {
            println!("cargo:warning=Successfully installed and using system GLPK");
            let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
            generate_bindings_simple(&out_dir);
            return;
        }
    }
    
    // Strategy 3: Build from source with better tooling
    println!("cargo:warning=Building GLPK from source...");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    
    if build_glpk_robust(&out_dir) {
        let glpk_lib_dir = out_dir.join("glpk-install/lib");
        println!("cargo:rustc-link-search=native={}", glpk_lib_dir.display());
        println!("cargo:rustc-link-lib=static=glpk");
        generate_bindings_simple(&out_dir);
    } else {
        panic!("Failed to build or install GLPK. Please install manually: brew install glpk (macOS) or sudo apt install libglpk-dev (Ubuntu)");
    }
}

fn try_system_glpk() -> bool {
    // First try with current PKG_CONFIG_PATH
    match pkg_config::Config::new().probe("glpk") {
        Ok(library) => {
            for path in &library.link_paths {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
            for lib in &library.libs {
                println!("cargo:rustc-link-lib={}", lib);
            }
            return true;
        },
        Err(_) => {
            // Try homebrew paths on macOS
            if std::env::consts::OS == "macos" {
                let homebrew_paths = [
                    "/opt/homebrew/lib/pkgconfig",  // Apple Silicon
                    "/usr/local/lib/pkgconfig",     // Intel Mac
                ];

                for path in &homebrew_paths {
                    if std::path::Path::new(path).join("glpk.pc").exists() {
                        // Update PKG_CONFIG_PATH with current path + homebrew path
                        let current_path = std::env::var("PKG_CONFIG_PATH").unwrap_or_default();
                        let new_path = if current_path.is_empty() {
                            path.to_string()
                        } else {
                            format!("{}:{}", current_path, path)
                        };
                        std::env::set_var("PKG_CONFIG_PATH", &new_path);

                        if let Ok(library) = pkg_config::Config::new().probe("glpk") {
                            for lib_path in &library.link_paths {
                                println!("cargo:rustc-link-search=native={}", lib_path.display());
                            }
                            for lib in &library.libs {
                                println!("cargo:rustc-link-lib={}", lib);
                            }
                            return true;
                        }
                    }
                }

                // Also try direct library detection if pkg-config fails
                let homebrew_lib_paths = [
                    "/opt/homebrew/lib",  // Apple Silicon
                    "/usr/local/lib",     // Intel Mac
                ];

                for lib_path in &homebrew_lib_paths {
                    let glpk_lib = std::path::Path::new(lib_path).join("libglpk.dylib");
                    if glpk_lib.exists() {
                        println!("cargo:rustc-link-search=native={}", lib_path);
                        println!("cargo:rustc-link-lib=glpk");
                        println!("cargo:warning=Found GLPK library at {}", glpk_lib.display());
                        return true;
                    }
                }
            }

            // Try direct library detection on Linux if pkg-config fails
            if std::env::consts::OS == "linux" {
                let linux_lib_paths = [
                    "/usr/lib/x86_64-linux-gnu",  // Ubuntu/Debian x64
                    "/usr/lib/aarch64-linux-gnu", // Ubuntu/Debian ARM64
                    "/usr/lib64",                  // RedHat/CentOS
                    "/usr/lib",                    // Generic
                ];

                for lib_path in &linux_lib_paths {
                    let glpk_lib = std::path::Path::new(lib_path).join("libglpk.so");
                    if glpk_lib.exists() {
                        println!("cargo:rustc-link-search=native={}", lib_path);
                        println!("cargo:rustc-link-lib=glpk");
                        println!("cargo:warning=Found GLPK library at {}", glpk_lib.display());
                        return true;
                    }
                }
            }
        }
    }

    false
}

fn try_install_system_glpk() -> bool {
    let os = std::env::consts::OS;
    
    match os {
        "macos" => {
            println!("cargo:warning=Attempting to install GLPK via Homebrew...");
            // Check if homebrew is available
            if Command::new("brew").arg("--version").output().is_ok() {
                let output = Command::new("brew")
                    .args(&["install", "glpk"])
                    .output();
                
                match output {
                    Ok(out) if out.status.success() => {
                        println!("cargo:warning=Successfully installed GLPK via Homebrew");
                        true
                    },
                    _ => {
                        println!("cargo:warning=Homebrew install failed, trying source build");
                        false
                    }
                }
            } else {
                println!("cargo:warning=Homebrew not found, trying source build");
                false
            }
        },
        "linux" => {
            println!("cargo:warning=Attempting to install GLPK via system package manager...");
            // Try apt first (Ubuntu/Debian)
            if Command::new("apt").arg("--version").output().is_ok() {
                // First update package list
                let update_result = Command::new("sudo")
                    .args(&["apt", "update"])
                    .status();
                
                if update_result.is_ok() {
                    // Then install GLPK
                    let output = Command::new("sudo")
                        .args(&["apt", "install", "-y", "libglpk-dev"])
                        .output();
                    
                    if let Ok(out) = output {
                        if out.status.success() {
                            println!("cargo:warning=Successfully installed GLPK via apt");
                            return true;
                        }
                    }
                }
            }
            
            // Try yum (RedHat/CentOS)
            if Command::new("yum").arg("--version").output().is_ok() {
                let output = Command::new("sudo")
                    .args(&["yum", "install", "-y", "glpk-devel"])
                    .output();
                
                if let Ok(out) = output {
                    if out.status.success() {
                        println!("cargo:warning=Successfully installed GLPK via yum");
                        return true;
                    }
                }
            }
            
            false
        },
        _ => {
            println!("cargo:warning=Unsupported OS for automatic installation, trying source build");
            false
        }
    }
}

fn build_glpk_robust(out_dir: &PathBuf) -> bool {
    let glpk_version = "5.0";
    let install_dir = out_dir.join("glpk-install");
    let src_dir = out_dir.join(format!("glpk-{}", glpk_version));
    
    // Create install directory
    std::fs::create_dir_all(&install_dir).ok();
    
    // Download and extract GLPK
    if !download_and_extract_glpk(out_dir, glpk_version) {
        return false;
    }
    
    // Try CMake build first (more robust)
    if try_cmake_build(&src_dir, &install_dir) {
        return true;
    }
    
    // Fallback to autotools build
    try_autotools_build(&src_dir, &install_dir)
}

fn download_and_extract_glpk(out_dir: &PathBuf, version: &str) -> bool {
    let tar_file = format!("glpk-{}.tar.gz", version);
    let tar_path = out_dir.join(&tar_file);
    let src_dir = out_dir.join(format!("glpk-{}", version));
    
    // Skip if already extracted
    if src_dir.exists() {
        return true;
    }
    
    // Download if not exists
    if !tar_path.exists() {
        let url = format!("https://ftp.gnu.org/gnu/glpk/{}", tar_file);
        println!("cargo:warning=Downloading GLPK from {}", url);
        
        // Use reqwest for reliable download
        if let Ok(response) = reqwest::blocking::get(&url) {
            if response.status().is_success() {
                if let Ok(bytes) = response.bytes() {
                    if std::fs::write(&tar_path, &bytes).is_ok() {
                        println!("cargo:warning=Downloaded GLPK successfully");
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
    
    // Extract
    println!("cargo:warning=Extracting GLPK...");
    let tar_file = match std::fs::File::open(&tar_path) {
        Ok(file) => file,
        Err(_) => return false,
    };
    let tar_decoder = flate2::read::GzDecoder::new(tar_file);
    let mut archive = tar::Archive::new(tar_decoder);
    
    match archive.unpack(out_dir) {
        Ok(_) => {
            println!("cargo:warning=Extracted GLPK successfully");
            true
        },
        Err(_) => false,
    }
}

fn try_cmake_build(src_dir: &PathBuf, install_dir: &PathBuf) -> bool {
    println!("cargo:warning=Trying CMake build...");
    
    // Check if CMakeLists.txt exists or create a simple one
    let cmake_file = src_dir.join("CMakeLists.txt");
    if !cmake_file.exists() {
        // GLPK doesn't come with CMake, so skip this method
        return false;
    }
    
    let _dst = cmake::Config::new(src_dir)
        .define("CMAKE_INSTALL_PREFIX", install_dir.to_str().unwrap())
        .define("BUILD_SHARED_LIBS", "OFF")
        .build();
    
    // Check if build was successful
    install_dir.join("lib").exists() && 
    (install_dir.join("lib/libglpk.a").exists() || install_dir.join("lib/glpk.lib").exists())
}

fn try_autotools_build(src_dir: &PathBuf, install_dir: &PathBuf) -> bool {
    println!("cargo:warning=Trying autotools build...");
    
    let configure_script = src_dir.join("configure");
    if !configure_script.exists() {
        return false;
    }
    
    // Configure
    let configure_result = Command::new("./configure")
        .args(&[
            &format!("--prefix={}", install_dir.display()),
            "--disable-shared",
            "--enable-static",
            "--disable-dependency-tracking",
        ])
        .current_dir(src_dir)
        .status();
    
    if configure_result.is_err() || !configure_result.unwrap().success() {
        return false;
    }
    
    // Build
    let make_result = Command::new("make")
        .args(&["-j", &num_cpus::get().to_string()])
        .current_dir(src_dir)
        .status();
    
    if make_result.is_err() || !make_result.unwrap().success() {
        return false;
    }
    
    // Install
    let install_result = Command::new("make")
        .arg("install")
        .current_dir(src_dir)
        .status();
    
    if install_result.is_err() || !install_result.unwrap().success() {
        return false;
    }
    
    // Verify installation
    install_dir.join("lib/libglpk.a").exists()
}

fn generate_bindings_simple(out_dir: &PathBuf) {
    let bindings_content = r#"
use libc::{c_int, c_double, c_char};

#[repr(C)]
pub struct glp_prob {
    _private: [u8; 0],
}

#[repr(C)]
pub struct glp_iocp {
    pub msg_lev: c_int,
    pub br_tech: c_int,
    pub bt_tech: c_int,
    pub tol_int: c_double,
    pub tol_obj: c_double,
    pub tm_lim: c_int,
    pub out_frq: c_int,
    pub out_dly: c_int,
    pub cb_func: *mut core::ffi::c_void,
    pub cb_info: *mut core::ffi::c_void,
    pub cb_size: c_int,
    pub pp_tech: c_int,
    pub mip_gap: c_double,
    pub mir_cuts: c_int,
    pub gmi_cuts: c_int,
    pub cov_cuts: c_int,
    pub clq_cuts: c_int,
    pub presolve: c_int,
    pub binarize: c_int,
    pub fp_heur: c_int,
    pub ps_heur: c_int,
    pub ps_tm_lim: c_int,
    pub sr_heur: c_int,
    pub use_sol: c_int,
    pub save_sol: *const c_char,
    pub alien: c_int,
    pub flip: c_int,
}

impl Default for glp_iocp {
    fn default() -> Self {
        unsafe { std::mem::zeroed() }
    }
}

extern "C" {
    pub fn glp_create_prob() -> *mut glp_prob;
    pub fn glp_delete_prob(lp: *mut glp_prob);
    pub fn glp_set_obj_dir(lp: *mut glp_prob, dir: c_int);
    pub fn glp_add_rows(lp: *mut glp_prob, nrs: c_int) -> c_int;
    pub fn glp_add_cols(lp: *mut glp_prob, ncs: c_int) -> c_int;
    pub fn glp_set_row_bnds(lp: *mut glp_prob, i: c_int, type_: c_int, lb: c_double, ub: c_double);
    pub fn glp_set_col_bnds(lp: *mut glp_prob, j: c_int, type_: c_int, lb: c_double, ub: c_double);
    pub fn glp_set_col_kind(lp: *mut glp_prob, j: c_int, kind: c_int);
    pub fn glp_set_obj_coef(lp: *mut glp_prob, j: c_int, coef: c_double);
    pub fn glp_load_matrix(lp: *mut glp_prob, ne: c_int, ia: *const c_int, ja: *const c_int, ar: *const c_double);
    pub fn glp_init_iocp(parm: *mut glp_iocp);
    pub fn glp_intopt(lp: *mut glp_prob, parm: *const glp_iocp) -> c_int;
    pub fn glp_mip_status(lp: *mut glp_prob) -> c_int;
    pub fn glp_mip_obj_val(lp: *mut glp_prob) -> c_double;
    pub fn glp_mip_col_val(lp: *mut glp_prob, j: c_int) -> c_double;
    pub fn glp_term_out(flag: c_int) -> c_int;
}
"#;
    
    let out_path = out_dir.join("bindings.rs");
    std::fs::write(out_path, bindings_content).expect("Couldn't write bindings!");
}