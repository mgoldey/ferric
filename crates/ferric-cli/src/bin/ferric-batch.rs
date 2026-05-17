use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!("Usage: ferric-batch <template.toml> <xyz_dir> <output_dir>");
        std::process::exit(1);
    }

    let template_path = &args[1];
    let xyz_dir = Path::new(&args[2]);
    let out_dir = Path::new(&args[3]);

    if !xyz_dir.is_dir() {
        eprintln!("Error: {} is not a directory", xyz_dir.display());
        std::process::exit(1);
    }

    if !out_dir.exists() {
        fs::create_dir_all(out_dir).unwrap_or_else(|e| {
            eprintln!("Failed to create output directory: {}", e);
            std::process::exit(1);
        });
    }

    let template = fs::read_to_string(template_path).unwrap_or_else(|e| {
        eprintln!("Failed to read template: {}", e);
        std::process::exit(1);
    });

    let entries = fs::read_dir(xyz_dir).unwrap_or_else(|e| {
        eprintln!("Failed to read xyz directory: {}", e);
        std::process::exit(1);
    });

    let mut success_count = 0;
    let mut fail_count = 0;

    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let path = entry.path();
        
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("xyz") {
            let file_stem = path.file_stem().unwrap().to_str().unwrap();
            
            // Create specific config for this molecule
            let config_content = template.replace("{XYZ_FILE}", path.to_str().unwrap());
            
            let toml_path = out_dir.join(format!("{}.toml", file_stem));
            let out_log_path = out_dir.join(format!("{}.out", file_stem));
            
            fs::write(&toml_path, config_content).unwrap();

            println!("Running {}...", file_stem);
            
            let current_exe = env::current_exe().unwrap_or_default();
            let ferric_bin = if let Some(parent) = current_exe.parent() {
                let bin = parent.join("ferric");
                if bin.exists() {
                    bin.into_os_string()
                } else {
                    "ferric".into()
                }
            } else {
                "ferric".into()
            };

            let status = Command::new(ferric_bin)
                .arg(&toml_path)
                .stdout(fs::File::create(&out_log_path).unwrap())
                .stderr(fs::File::create(out_dir.join(format!("{}.err", file_stem))).unwrap())
                .status();

            match status {
                Ok(s) if s.success() => {
                    println!("  Success.");
                    success_count += 1;
                }
                _ => {
                    println!("  Failed. Check {}.err", file_stem);
                    fail_count += 1;
                }
            }
        }
    }

    println!("\nBatch run complete.");
    println!("Successful: {}", success_count);
    println!("Failed:     {}", fail_count);
}
