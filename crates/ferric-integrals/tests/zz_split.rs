use ferric_integrals::ffi;
#[test]
#[ignore]
fn t() {
    for s in [0.5_f64, 20.0] {
        let mut o = [0.0f64; 6];
        let i = unsafe { ffi::scf_terf_delta_split(8, 20_000, s, o.as_mut_ptr()) };
        eprintln!("\n  s={s}  I={i}");
        let lbl = ["TOTAL", "tail build", "df build", "pmf build", "m-ladder+dots", "boys_upto"];
        for (l, v) in lbl.iter().zip(&o) {
            eprintln!("    {l:16} {v:8.1} ns  ({:5.1}%)", 100.0*v/o[0].max(1e-9));
        }
    }
}
