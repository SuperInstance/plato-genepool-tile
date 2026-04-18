//! plato-genepool-tile — lossless bridge between cuda-genepool Gene and Plato Tile.
//!
//! GeneTile is a *union* struct: every Gene field stored verbatim alongside its
//! tile-spec mirror, so gene_to_tile → tile_to_gene is a perfect round-trip.

use std::time::{SystemTime, UNIX_EPOCH};
fn now_ns() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64) }
fn gen_id(p: &str) -> String { format!("{}-{:x}", p, now_ns() & 0xFFFF_FFFF) }

// ─── Enums ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Instinct { Perceive, Navigate, Survive, Communicate, Learn, Share, Rest, Explore, Defend, Cooperate }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GeneCategory {
    Algorithm, DataStructure, Optimization, ErrorHandling, Security, UI,
    Integration, Testing, Perception, Navigation, Survival, Communication, Learning,
}

fn instinct_str(i: Instinct) -> &'static str {
    match i { Instinct::Perceive=>"perceive",Instinct::Navigate=>"navigate",Instinct::Survive=>"survive",
        Instinct::Communicate=>"communicate",Instinct::Learn=>"learn",Instinct::Share=>"share",
        Instinct::Rest=>"rest",Instinct::Explore=>"explore",Instinct::Defend=>"defend",
        Instinct::Cooperate=>"cooperate" }
}
fn category_str(c: GeneCategory) -> &'static str {
    match c { GeneCategory::Algorithm=>"algorithm",GeneCategory::DataStructure=>"data_structure",
        GeneCategory::Optimization=>"optimization",GeneCategory::ErrorHandling=>"error_handling",
        GeneCategory::Security=>"security",GeneCategory::UI=>"ui",GeneCategory::Integration=>"integration",
        GeneCategory::Testing=>"testing",GeneCategory::Perception=>"perception",
        GeneCategory::Navigation=>"navigation",GeneCategory::Survival=>"survival",
        GeneCategory::Communication=>"communication",GeneCategory::Learning=>"learning" }
}
fn category_instinct(c: GeneCategory) -> Option<Instinct> {
    match c { GeneCategory::Perception=>Some(Instinct::Perceive),GeneCategory::Navigation=>Some(Instinct::Navigate),
        GeneCategory::Survival=>Some(Instinct::Survive),GeneCategory::Communication=>Some(Instinct::Communicate),
        GeneCategory::Learning=>Some(Instinct::Learn),GeneCategory::Integration=>Some(Instinct::Cooperate),
        GeneCategory::Security=>Some(Instinct::Defend),_=>None }
}
fn instinct_energy(i: Instinct) -> f64 {
    match i { Instinct::Perceive=>0.02,Instinct::Navigate=>0.05,Instinct::Survive=>0.01,
        Instinct::Communicate=>0.03,Instinct::Learn=>0.04,Instinct::Share=>0.02,
        Instinct::Rest=>0.001,Instinct::Explore=>0.06,Instinct::Defend=>0.04,Instinct::Cooperate=>0.02 }
}

// ─── Gene ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Gene {
    pub id: String, pub pattern: String, pub category: GeneCategory,
    pub fitness: f64, pub energy_cost: f64, pub expression_level: f64, pub dominance: f64,
    pub usage_count: u64, pub success_count: u64, pub failure_count: u64,
    pub generation: u32, pub origin_agent: String, pub parents: Vec<String>,
    pub quarantined: bool, pub quarantine_reason: Option<String>, pub instinct: Option<Instinct>,
}
impl Gene {
    pub fn new(pattern: &str, category: GeneCategory, fitness: f64) -> Self {
        let instinct = category_instinct(category);
        Self { id: gen_id("gene"), pattern: pattern.to_string(), category,
            fitness: fitness.clamp(0.0, 1.0), energy_cost: instinct.map_or(0.03, instinct_energy),
            expression_level: fitness.clamp(0.1, 1.0), dominance: 0.5,
            usage_count: 0, success_count: 0, failure_count: 0, generation: 0,
            origin_agent: String::new(), parents: vec![],
            quarantined: false, quarantine_reason: None, instinct }
    }
}

// ─── GeneTile ─────────────────────────────────────────────────────────────────

/// Union of Gene + Tile fields. All gene fields stored verbatim → lossless round-trip.
#[derive(Debug, Clone)]
pub struct GeneTile {
    // Gene fields (verbatim)
    pub gene_id: String,    pub pattern: String,   pub category: GeneCategory,
    pub fitness: f64,       pub energy_cost: f64,  pub expression_level: f64,
    pub dominance: f64,     pub usage_count: u64,  pub success_count: u64,
    pub failure_count: u64, pub generation: u32,   pub origin_agent: String,
    pub parents: Vec<String>, pub quarantined: bool, pub quarantine_reason: Option<String>,
    pub instinct: Option<Instinct>,
    // Tile fields (derived from gene semantics)
    pub tile_id: String,          pub confidence: f64,         pub question: String,
    pub answer: String,           pub tags: Vec<String>,       pub anchors: Vec<String>,
    pub weight: f64,              pub use_count: u64,          pub active: bool,
    pub last_used_tick: u64,      pub constraint_tolerance: f64, pub constraint_threshold: f64,
    // Bridge
    pub conversion_score: f64,    pub activation_count: u64,
}

/// Standalone Tile with gene metadata stripped — result of GeneTilePool::harvest.
#[derive(Debug, Clone)]
pub struct Tile {
    pub id: String, pub confidence: f64, pub question: String, pub answer: String,
    pub tags: Vec<String>, pub anchors: Vec<String>, pub weight: f64,
    pub use_count: u64, pub active: bool, pub last_used_tick: u64,
    pub constraint_tolerance: f64, pub constraint_threshold: f64,
}

// ─── Conversion ───────────────────────────────────────────────────────────────

/// Gene → GeneTile. All gene fields preserved; tile fields derived from gene semantics.
pub fn gene_to_tile(g: &Gene) -> GeneTile {
    let mut tags = vec![category_str(g.category).to_string()];
    if let Some(i) = g.instinct { tags.push(instinct_str(i).to_string()); }
    let cs = (g.fitness * g.expression_level
        + if g.instinct.is_some() { 0.1 } else { 0.0 }
        - if g.quarantined { 0.3 } else { 0.0 }).clamp(0.0, 1.0);
    GeneTile {
        gene_id: g.id.clone(), pattern: g.pattern.clone(), category: g.category,
        fitness: g.fitness, energy_cost: g.energy_cost, expression_level: g.expression_level,
        dominance: g.dominance, usage_count: g.usage_count, success_count: g.success_count,
        failure_count: g.failure_count, generation: g.generation,
        origin_agent: g.origin_agent.clone(), parents: g.parents.clone(),
        quarantined: g.quarantined, quarantine_reason: g.quarantine_reason.clone(),
        instinct: g.instinct, tile_id: gen_id("tile"), confidence: g.fitness,
        question: g.pattern.clone(), answer: format!("[{}] {}", category_str(g.category), g.pattern),
        tags, anchors: g.parents.clone(), weight: g.expression_level,
        use_count: g.usage_count, active: !g.quarantined, last_used_tick: 0,
        constraint_tolerance: 0.05, constraint_threshold: g.fitness.max(0.1),
        conversion_score: cs, activation_count: 0,
    }
}

/// GeneTile → Gene. Exact inverse — lossless by construction.
pub fn tile_to_gene(gt: &GeneTile) -> Gene {
    Gene { id: gt.gene_id.clone(), pattern: gt.pattern.clone(), category: gt.category,
        fitness: gt.fitness, energy_cost: gt.energy_cost, expression_level: gt.expression_level,
        dominance: gt.dominance, usage_count: gt.usage_count, success_count: gt.success_count,
        failure_count: gt.failure_count, generation: gt.generation,
        origin_agent: gt.origin_agent.clone(), parents: gt.parents.clone(),
        quarantined: gt.quarantined, quarantine_reason: gt.quarantine_reason.clone(),
        instinct: gt.instinct }
}

pub fn batch_convert(genes: &[Gene]) -> Vec<GeneTile> { genes.iter().map(gene_to_tile).collect() }

// ─── GeneTilePool ─────────────────────────────────────────────────────────────

pub struct GeneTilePool { tiles: Vec<GeneTile>, generation: u32 }

impl Default for GeneTilePool { fn default() -> Self { Self::new() } }

impl GeneTilePool {
    pub fn new() -> Self { Self { tiles: vec![], generation: 0 } }
    pub fn insert(&mut self, gt: GeneTile) { self.tiles.push(gt); }

    pub fn best_for_query(&self, query: &str) -> Option<&GeneTile> {
        let q = query.to_lowercase();
        self.tiles.iter()
            .filter(|gt| gt.active && !gt.quarantined
                && (gt.tags.iter().any(|t| q.contains(t.as_str()) || t.contains(q.as_str()))
                    || gt.pattern.to_lowercase().contains(&q)))
            .max_by(|a, b| (a.fitness * a.expression_level)
                .partial_cmp(&(b.fitness * b.expression_level))
                .unwrap_or(std::cmp::Ordering::Equal))
    }

    /// One generation: crossover top-2, drift expression toward fitness, prune bottom quartile.
    pub fn evolve(&mut self) {
        self.generation += 1;
        let top: Vec<GeneTile> = {
            let mut v: Vec<&GeneTile> = self.tiles.iter()
                .filter(|gt| gt.active && !gt.quarantined).collect();
            v.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
            v.iter().take(2).map(|gt| (*gt).clone()).collect()
        };
        if top.len() >= 2 { self.tiles.push(crossover(&top[0], &top[1], self.generation)); }
        let gen = self.generation;
        for gt in &mut self.tiles {
            if !gt.quarantined {
                gt.expression_level = (gt.expression_level + (gt.fitness - gt.expression_level) * 0.05)
                    .clamp(0.05, 1.0);
                gt.weight = gt.expression_level; gt.generation = gen;
            }
        }
        let n = self.tiles.len();
        if n > 4 {
            let mut ord: Vec<usize> = (0..n).collect();
            ord.sort_by(|&a, &b| self.tiles[a].fitness.partial_cmp(&self.tiles[b].fitness)
                .unwrap_or(std::cmp::Ordering::Equal));
            for &i in ord.iter().take(n / 4) { self.tiles[i].active = false; }
        }
    }

    pub fn quarantine_below(&mut self, threshold: f64) -> usize {
        let mut n = 0;
        for gt in &mut self.tiles {
            if gt.fitness < threshold && !gt.quarantined {
                gt.quarantined = true; gt.active = false; n += 1;
                gt.quarantine_reason = Some(format!("fitness {:.3} < {:.3}", gt.fitness, threshold));
            }
        }
        n
    }

    pub fn harvest(&self, n: usize) -> Vec<Tile> {
        let mut v: Vec<&GeneTile> = self.tiles.iter().filter(|gt| gt.active && !gt.quarantined).collect();
        v.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
        v.iter().take(n).map(|gt| Tile {
            id: gt.tile_id.clone(), confidence: gt.confidence, question: gt.question.clone(),
            answer: gt.answer.clone(), tags: gt.tags.clone(), anchors: gt.anchors.clone(),
            weight: gt.weight, use_count: gt.use_count, active: gt.active,
            last_used_tick: gt.last_used_tick, constraint_tolerance: gt.constraint_tolerance,
            constraint_threshold: gt.constraint_threshold,
        }).collect()
    }

    pub fn len(&self) -> usize { self.tiles.len() }
    pub fn is_empty(&self) -> bool { self.tiles.is_empty() }
    pub fn active_count(&self) -> usize { self.tiles.iter().filter(|gt| gt.active).count() }
    pub fn generation(&self) -> u32 { self.generation }
}

fn crossover(a: &GeneTile, b: &GeneTile, gen: u32) -> GeneTile {
    let mut tags = a.tags.clone();
    for t in &b.tags { if !tags.contains(t) { tags.push(t.clone()); } }
    let f = (a.fitness + b.fitness) / 2.0;
    GeneTile { gene_id: gen_id("gene"), tile_id: gen_id("tile"),
        pattern: a.pattern.clone(), category: a.category, fitness: f,
        energy_cost: (a.energy_cost + b.energy_cost) / 2.0,
        expression_level: (a.expression_level + b.expression_level) / 2.0,
        dominance: (a.dominance + b.dominance) / 2.0,
        usage_count: 0, success_count: 0, failure_count: 0, generation: gen,
        origin_agent: a.origin_agent.clone(), parents: vec![a.gene_id.clone(), b.gene_id.clone()],
        quarantined: false, quarantine_reason: None, instinct: a.instinct,
        confidence: f, question: a.question.clone(), answer: a.answer.clone(), tags,
        anchors: a.anchors.clone(), weight: (a.weight + b.weight) / 2.0,
        use_count: 0, active: true, last_used_tick: 0,
        constraint_tolerance: (a.constraint_tolerance + b.constraint_tolerance) / 2.0,
        constraint_threshold: (a.constraint_threshold + b.constraint_threshold) / 2.0,
        conversion_score: (a.conversion_score + b.conversion_score) / 2.0, activation_count: 0 }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn nav() -> Gene {
        let mut g = Gene::new("navigate_path", GeneCategory::Navigation, 0.8);
        g.origin_agent = "scout".into();
        g.usage_count = 10; g.success_count = 8; g.failure_count = 2; g
    }
    fn pool4() -> GeneTilePool {
        let mut p = GeneTilePool::new();
        p.insert(gene_to_tile(&Gene::new("survive",      GeneCategory::Survival,   0.9)));
        p.insert(gene_to_tile(&Gene::new("navigate",     GeneCategory::Navigation, 0.8)));
        p.insert(gene_to_tile(&Gene::new("learn_pattern",GeneCategory::Learning,   0.6)));
        p.insert(gene_to_tile(&Gene::new("bad_pattern",  GeneCategory::Algorithm,  0.2)));
        p
    }

    #[test] fn test_conversion_fitness_to_confidence() {
        let g = nav(); assert!((gene_to_tile(&g).confidence - g.fitness).abs() < f64::EPSILON);
    }
    #[test] fn test_conversion_pattern_to_question() {
        assert_eq!(gene_to_tile(&nav()).question, nav().pattern);
    }
    #[test] fn test_conversion_quarantine_to_active() {
        let mut g = nav(); g.quarantined = true;
        assert!(!gene_to_tile(&g).active); assert!(gene_to_tile(&nav()).active);
    }
    #[test] fn test_conversion_instinct_in_tags() {
        let gt = gene_to_tile(&Gene::new("nav", GeneCategory::Navigation, 0.8));
        assert!(gt.tags.contains(&"navigate".to_string()) && gt.tags.contains(&"navigation".to_string()));
    }
    #[test] fn test_roundtrip_fitness() {
        let g = nav(); assert!((g.fitness - tile_to_gene(&gene_to_tile(&g)).fitness).abs() < f64::EPSILON);
    }
    #[test] fn test_roundtrip_energy_cost() {
        let g = nav(); assert!((g.energy_cost - tile_to_gene(&gene_to_tile(&g)).energy_cost).abs() < f64::EPSILON);
    }
    #[test] fn test_roundtrip_instinct() {
        let g = nav(); assert_eq!(g.instinct, tile_to_gene(&gene_to_tile(&g)).instinct);
    }
    #[test] fn test_roundtrip_quarantine() {
        let mut g = nav(); g.quarantined = true; g.quarantine_reason = Some("bad actor".into());
        let g2 = tile_to_gene(&gene_to_tile(&g));
        assert!(g2.quarantined); assert_eq!(g2.quarantine_reason, Some("bad actor".into()));
    }
    #[test] fn test_roundtrip_counts() {
        let g2 = tile_to_gene(&gene_to_tile(&nav()));
        assert_eq!((g2.usage_count, g2.success_count, g2.failure_count), (10, 8, 2));
    }
    #[test] fn test_batch_convert() {
        let gs = vec![Gene::new("a",GeneCategory::Algorithm,0.7),
                      Gene::new("b",GeneCategory::Security,0.8),
                      Gene::new("c",GeneCategory::Navigation,0.9)];
        let ts = batch_convert(&gs);
        assert_eq!(ts.len(), 3); assert!((ts[2].fitness - 0.9).abs() < f64::EPSILON);
    }
    #[test] fn test_pool_insert() {
        let mut p = GeneTilePool::new(); p.insert(gene_to_tile(&nav()));
        assert_eq!(p.len(), 1); assert_eq!(p.active_count(), 1);
    }
    #[test] fn test_pool_best_for_query() {
        let p = pool4();
        let gt = p.best_for_query("navigate").unwrap();
        assert!(gt.tags.contains(&"navigate".to_string()) || gt.pattern.contains("navigate"));
        // highest fitness wins among matches
        let mut p2 = GeneTilePool::new();
        p2.insert(gene_to_tile(&Gene::new("survive_lo", GeneCategory::Survival, 0.5)));
        p2.insert(gene_to_tile(&Gene::new("survive_hi", GeneCategory::Survival, 0.95)));
        assert!(p2.best_for_query("survive").unwrap().fitness >= 0.95 - f64::EPSILON);
    }
    #[test] fn test_pool_quarantine_below() {
        let mut p = pool4(); let count = p.quarantine_below(0.5);
        assert!(count > 0);
        for gt in p.tiles.iter() {
            if gt.fitness < 0.5 { assert!(gt.quarantined && !gt.active); } else { assert!(!gt.quarantined); }
        }
    }
    #[test] fn test_pool_evolve() {
        let mut p = pool4(); let before = p.len(); p.evolve();
        assert_eq!(p.generation(), 1); assert!(p.len() > before);
        let child = &p.tiles[before];
        assert_eq!(child.generation, 1);
        assert!((child.fitness - (0.9 + 0.8) / 2.0).abs() < 1e-10);
    }
    #[test] fn test_pool_harvest() {
        let p = pool4(); let tiles = p.harvest(2);
        assert_eq!(tiles.len(), 2);
        assert!(!tiles[0].id.is_empty() && tiles[0].confidence >= tiles[1].confidence);
    }
    #[test] fn test_conversion_score_range() {
        let cases = [Gene::new("a", GeneCategory::Survival, 1.0),
                     Gene::new("b", GeneCategory::Algorithm, 0.0),
                     { let mut g = Gene::new("c", GeneCategory::Navigation, 0.5); g.quarantined = true; g }];
        for gene in &cases {
            let s = gene_to_tile(gene).conversion_score;
            assert!((0.0..=1.0).contains(&s), "score {s} out of range");
        }
    }
}
