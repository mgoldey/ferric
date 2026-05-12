use std::collections::HashMap;
use std::sync::LazyLock;

static SYMBOL_TO_Z: LazyLock<HashMap<&'static str, i32>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    for &(sym, z) in ELEMENTS.iter() {
        m.insert(sym, z);
    }
    m
});

static Z_TO_SYMBOL: LazyLock<HashMap<i32, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    for &(sym, z) in ELEMENTS.iter() {
        m.insert(z, sym);
    }
    m
});

const ELEMENTS: &[(&str, i32)] = &[
    ("H", 1), ("He", 2), ("Li", 3), ("Be", 4), ("B", 5),
    ("C", 6), ("N", 7), ("O", 8), ("F", 9), ("Ne", 10),
    ("Na", 11), ("Mg", 12), ("Al", 13), ("Si", 14), ("P", 15),
    ("S", 16), ("Cl", 17), ("Ar", 18), ("K", 19), ("Ca", 20),
];

pub fn symbol_to_z(sym: &str) -> Option<i32> {
    let normalized: String = sym.chars().enumerate().map(|(i, c)| {
        if i == 0 { c.to_ascii_uppercase() } else { c.to_ascii_lowercase() }
    }).collect();
    SYMBOL_TO_Z.get(normalized.as_str()).copied()
}

pub fn z_to_symbol(z: i32) -> Option<&'static str> {
    Z_TO_SYMBOL.get(&z).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_to_z() {
        assert_eq!(symbol_to_z("H"), Some(1));
        assert_eq!(symbol_to_z("h"), Some(1));
        assert_eq!(symbol_to_z("O"), Some(8));
        assert_eq!(symbol_to_z("Cl"), Some(17));
        assert_eq!(symbol_to_z("CL"), Some(17));
        assert_eq!(symbol_to_z("Xx"), None);
    }

    #[test]
    fn test_z_to_symbol() {
        assert_eq!(z_to_symbol(1), Some("H"));
        assert_eq!(z_to_symbol(8), Some("O"));
        assert_eq!(z_to_symbol(999), None);
    }
}
