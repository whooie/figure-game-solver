#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(clippy::needless_return)]

use std::path::PathBuf;
use figure::{
    game::{
        Board,
        Config,
    },
    evolve::{
        Population,
        Probability,
    },
    solve::{
        Solver,
        Strategy,
    },
};

const POPULATION_SIZE: usize = 500;
const MUTATE: f64 = 0.8;

fn main() {
    let config = Config::from_file(PathBuf::from("config.toml"))
        .expect("problem reading config file");
    let board = Board::from_iter_shape(config.blocks.clone(), (5, 5));
    println!("{board}");

    let gen = || Strategy::new_random(&board, config.n_moves);
    let mut solver = Solver::init(POPULATION_SIZE, gen);
    let mutate: Probability = MUTATE.try_into()
        .expect("probability must be between 0 and 1");
    let stop = |solver: &Solver| {
        solver.get_strategies().iter().any(|s| s.get_score().unwrap() == 0)
    };
    let k: usize = solver.evolve_until(mutate, stop)
        .expect("failed to evolve population");

    println!("\n{k} generations");
    let strategy: &Strategy = solver.get_most_fit().unwrap();
    println!(
        "{} ({})\n{}",
        strategy,
        strategy.get_score().unwrap(),
        strategy.final_board().unwrap(),
    );
}
