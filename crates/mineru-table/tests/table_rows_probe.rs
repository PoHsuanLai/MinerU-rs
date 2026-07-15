//! Diagnostic probe: dumps what the real table models see on a real page crop.
//!
//! Reads a table crop PNG from `MINERU_TABLE_PROBE_IMG` and prints the wired/
//! wireless classification plus the SLANet structure-token stream (`<tr>`/`<td>`
//! counts). Used to locate where predicted rows diverge from the Python
//! reference; not a pass/fail gate.
//!
//! ```text
//! MINERU_MODELS_DIR=... MINERU_TABLE_PROBE_IMG=/path/p8_o9.png \
//!   cargo test -p mineru-table --test table_rows_probe -- --ignored --nocapture
//! ```

use mineru_burn_common::backend::Cpu;

#[test]
#[ignore = "requires MINERU_MODELS_DIR and MINERU_TABLE_PROBE_IMG; slow CPU forward"]
fn probe_table_crop() {
    let Ok(path) = std::env::var("MINERU_TABLE_PROBE_IMG") else {
        eprintln!("set MINERU_TABLE_PROBE_IMG to a table crop PNG");
        return;
    };
    let img = image::open(&path)
        .expect("probe image should open")
        .to_rgb8();
    println!("== {path}  ({}x{})", img.width(), img.height());

    let cls = mineru_table::cls::classify::<Cpu>(&img).expect("classifier should run");
    println!("classification: {:?} score={:.4}", cls.class, cls.score);
    let pre_cls = mineru_table::cls::preprocess(&img).expect("cls preprocess should run");
    let probs = mineru_table::cls::debug_forward::<Cpu>(pre_cls).expect("cls forward should run");
    println!("cls probs: wired={:.4} wireless={:.4}", probs[0], probs[1]);

    let root = std::env::var("MINERU_MODELS_DIR").expect("set MINERU_MODELS_DIR");
    let weights = std::path::Path::new(&root).join("TabRec/SlanetPlus/slanet-plus.onnx");
    let model = mineru_table::slanet::model::SlaNet::<Cpu>::load(&weights).expect("slanet should load");
    let pre = mineru_table::slanet::preprocess(&img);
    let raw = model.forward(&pre).expect("slanet forward should run");
    let vocab = mineru_table::slanet::build_vocab();
    let structure = mineru_table::slanet::decode::decode(&raw.as_preds(), &vocab, pre.orig_w, pre.orig_h);

    let tr = structure.tokens.iter().filter(|t| *t == "<tr>").count();
    let td = structure
        .tokens
        .iter()
        .filter(|t| t.starts_with("<td"))
        .count();
    println!(
        "decode steps={} tokens={} <tr>={tr} <td>={td} cell_bboxes={} score={:.4}",
        raw.len,
        structure.tokens.len(),
        structure.cell_bboxes.len(),
        structure.score
    );
    println!("tokens: {}", structure.tokens.join(""));
}
