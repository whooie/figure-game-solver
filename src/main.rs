#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(clippy::needless_return)]

use std::path::PathBuf;
use anyhow::Result;
use figure::{
    game::{
        self,
        Board,
    },
    evolve::{
        self,
        Population,
    },
    solve::{
        Solver,
        Strategy,
    },
};

fn main() -> Result<()> {
    let game_config = game::Config::from_file(PathBuf::from("config.toml"))?;
    let evolve_config = evolve::Config::from_file(PathBuf::from("config.toml"))?;
    let board = Board::from_iter_shape(game_config.blocks.clone(), (5, 5));
    println!("{board}");

    let gen = || Strategy::new_random(&board, game_config.n_moves);
    let mut solver = Solver::init(evolve_config.n_pop, gen);
    let stop = |solver: &Solver| {
        solver.get_strategies().iter().any(|s| s.get_score().unwrap() == 0)
    };
    let k: usize = solver.evolve_until(evolve_config.mutate, stop)?;

    println!("");
    println!("population size: {}", evolve_config.n_pop);
    println!("mutation probability: {}", evolve_config.mutate.p());
    println!("=> {k} generations");
    let strategy: &Strategy = solver.get_most_fit().unwrap();
    println!(
        "{}\n{}",
        strategy,
        strategy.final_board().unwrap(),
    );
    return Ok(());
}
