//! `ferric_cli::try_run` runs the CLI in-process and RETURNS its errors: the
//! variant fixes the exit status and `stderr_text()` is exactly what the
//! `ferric` binary prints for it (`ferric_cli::run` reports it and exits).
//! These cases fail before any calculation, so no process-wide calculation
//! state is installed; the last one loads a config first.

use ferric_cli::error::CliError;
use ferric_cli::try_run;

fn argv(args: &[&str]) -> Vec<String> {
    std::iter::once("ferric")
        .chain(args.iter().copied())
        .map(String::from)
        .collect()
}

#[test]
fn no_arguments_is_a_usage_error_carrying_the_help_text() {
    let e = try_run(argv(&[])).unwrap_err();
    assert!(matches!(e, CliError::Usage(_)), "{e:?}");
    assert_eq!(e.exit_code(), 2);
    let text = e.stderr_text();
    assert!(
        text.starts_with(
            "usage: ferric [--verbose|-v] [--json <path>|--no-json] [--gpu <preset>] <input.toml>\n\
             \x20      ferric --version|-V\n\n"
        ),
        "{text}"
    );
    assert!(
        text.contains("  --gpu <preset>  off, auto, on, mixed or auto-mixed"),
        "{text}"
    );
    assert!(
        text.ends_with("flag, build profile, libint version and limits) and exit.\n"),
        "{text}"
    );
}

#[test]
fn a_second_positional_argument_is_the_same_usage_error() {
    let e = try_run(argv(&["a.toml", "b.toml"])).unwrap_err();
    assert_eq!(e.exit_code(), 2);
    assert_eq!(
        e.stderr_text(),
        try_run(argv(&[])).unwrap_err().stderr_text()
    );
}

#[test]
fn flags_without_an_input_print_only_the_synopsis() {
    let e = try_run(argv(&["-v", "--no-json"])).unwrap_err();
    assert!(matches!(e, CliError::Usage(_)), "{e:?}");
    assert_eq!(e.exit_code(), 2);
    assert_eq!(
        e.stderr_text(),
        "usage: ferric [--verbose|-v] [--json <path>|--no-json] [--gpu <preset>] <input.toml>\n"
    );
}

#[test]
fn json_without_a_path_is_a_usage_error() {
    let e = try_run(argv(&["in.toml", "--json"])).unwrap_err();
    assert!(matches!(e, CliError::Usage(_)), "{e:?}");
    assert_eq!(e.exit_code(), 2);
    assert_eq!(
        e.stderr_text(),
        "error: --json requires a path (use --no-json to disable the run log)\n"
    );
}

#[test]
fn a_bad_gpu_preset_is_a_usage_error() {
    let e = try_run(argv(&["--gpu", "fast", "in.toml"])).unwrap_err();
    assert!(matches!(e, CliError::Usage(_)), "{e:?}");
    assert_eq!(e.exit_code(), 2);
    assert_eq!(
        e.stderr_text(),
        "error: --gpu: invalid GPU preset \"fast\" (expected off, auto, on, mixed or auto-mixed)\n"
    );
}

#[test]
fn version_and_help_return_ok() {
    assert!(try_run(argv(&["--version"])).is_ok());
    assert!(try_run(argv(&["-V"])).is_ok());
    assert!(try_run(argv(&["--help"])).is_ok());
    assert!(try_run(argv(&["-h"])).is_ok());
}

#[test]
fn a_missing_input_file_is_an_input_error() {
    let e = try_run(argv(&["/nonexistent-dir/in.toml"])).unwrap_err();
    assert!(matches!(e, CliError::Msg(_)), "{e:?}");
    assert_eq!(e.exit_code(), 1);
    let text = e.stderr_text();
    assert!(
        text.starts_with("error: cannot read config file \"/nonexistent-dir/in.toml\": "),
        "{text}"
    );
    assert_eq!(text, format!("error: {e}\n"));
}

#[test]
fn an_unknown_method_kind_is_refused_after_the_config_loads() {
    let water = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../testdata/molecules/water.xyz"
    );
    let dir = std::env::temp_dir().join(format!("ferric_try_run_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let toml = dir.join("in.toml");
    std::fs::write(
        &toml,
        format!(
            "[molecule]\nxyz = \"{water}\"\n[basis]\nname = \"sto-3g\"\n\
             [method]\nkind = \"hf-plus\"\n"
        ),
    )
    .unwrap();
    let r = try_run(argv(&[toml.to_str().unwrap(), "--no-json"]));
    let _ = std::fs::remove_dir_all(&dir);
    let e = r.unwrap_err();
    assert!(matches!(e, CliError::Msg(_)), "{e:?}");
    assert_eq!(e.exit_code(), 1);
    assert_eq!(
        e.stderr_text(),
        format!(
            "error: {}\n",
            ferric_cli::unsupported_method_message("hf-plus")
        )
    );
}
