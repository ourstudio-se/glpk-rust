use std::path::PathBuf;
use std::process::Command;
use std::env;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let glpk_version = "5.0";
    let glpk_dir = out_dir.join(format!("glpk-{}", glpk_version));
    let glpk_lib_dir = glpk_dir.join("lib");
    let glpk_include_dir = glpk_dir.join("include");
    
    // Check if GLPK is already built
    if !glpk_lib_dir.join("libglpk.a").exists() {
        println!("cargo:warning=Building GLPK {} from source...", glpk_version);
        build_glpk(&out_dir, glpk_version);
    }
    
    // Tell cargo to link against our static library
    println!("cargo:rustc-link-search=native={}", glpk_lib_dir.display());
    println!("cargo:rustc-link-lib=static=glpk");
    
    // Tell cargo to include our headers
    println!("cargo:include={}", glpk_include_dir.display());
    
    // Generate bindings
    generate_bindings(&glpk_include_dir, &out_dir);
}

fn build_glpk(out_dir: &PathBuf, version: &str) {
    let glpk_tar = format!("glpk-{}.tar.gz", version);
    let glpk_url = format!("https://ftp.gnu.org/gnu/glpk/{}", glpk_tar);
    let glpk_dir = out_dir.join(format!("glpk-{}", version));
    let glpk_src_dir = out_dir.join(format!("glpk-{}", version));
    
    // Download GLPK if not already downloaded
    let tar_path = out_dir.join(&glpk_tar);
    if !tar_path.exists() {
        println!("cargo:warning=Downloading GLPK from {}", glpk_url);
        let output = Command::new("curl")
            .args(&["-L", "-o", tar_path.to_str().unwrap(), &glpk_url])
            .output()
            .expect("Failed to download GLPK. Make sure curl is installed.");
        
        if !output.status.success() {
            panic!("Failed to download GLPK: {}", String::from_utf8_lossy(&output.stderr));
        }
    }
    
    // Extract GLPK
    if !glpk_src_dir.exists() {
        println!("cargo:warning=Extracting GLPK...");
        let output = Command::new("tar")
            .args(&["-xzf", tar_path.to_str().unwrap()])
            .current_dir(out_dir)
            .output()
            .expect("Failed to extract GLPK. Make sure tar is installed.");
        
        if !output.status.success() {
            panic!("Failed to extract GLPK: {}", String::from_utf8_lossy(&output.stderr));
        }
    }
    
    // Configure GLPK
    let configure_script = glpk_src_dir.join("configure");
    if configure_script.exists() {
        println!("cargo:warning=Configuring GLPK...");
        let output = Command::new("./configure")
            .args(&[
                &format!("--prefix={}", glpk_dir.display()),
                "--disable-shared",
                "--enable-static",
                "--disable-dependency-tracking",
            ])
            .current_dir(&glpk_src_dir)
            .output()
            .expect("Failed to configure GLPK");
        
        if !output.status.success() {
            panic!("Failed to configure GLPK: {}", String::from_utf8_lossy(&output.stderr));
        }
    }
    
    // Build GLPK
    println!("cargo:warning=Building GLPK...");
    let output = Command::new("make")
        .args(&["-j", &num_cpus::get().to_string()])
        .current_dir(&glpk_src_dir)
        .output()
        .expect("Failed to build GLPK. Make sure make is installed.");
    
    if !output.status.success() {
        panic!("Failed to build GLPK: {}", String::from_utf8_lossy(&output.stderr));
    }
    
    // Install GLPK
    println!("cargo:warning=Installing GLPK...");
    let output = Command::new("make")
        .args(&["install"])
        .current_dir(&glpk_src_dir)
        .output()
        .expect("Failed to install GLPK");
    
    if !output.status.success() {
        panic!("Failed to install GLPK: {}", String::from_utf8_lossy(&output.stderr));
    }
}

fn generate_bindings(_include_dir: &PathBuf, out_dir: &PathBuf) {
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