#![allow(dead_code, non_snake_case, non_upper_case_globals)]
#![allow(clippy::needless_return)]

use figure::{
    game::Board,
    evolve::{
        Population,
        Probability,
    },
    solve::{
        Solver,
        Strategy,
    },
};

const BOARD: [[u8; 5]; 5] = [
    [ 2, 2, 3, 1, 4 ],
    [ 4, 2, 1, 3, 4 ],
    [ 2, 3, 2, 4, 2 ],
    [ 2, 4, 1, 1, 4 ],
    [ 1, 4, 4, 4, 1 ],
];
const MOVES: usize = 10;
const POPULATION_SIZE: usize = 500;
const MUTATE: f64 = 0.8;

fn main() {
    let board = Board::from(BOARD);
    println!("{board}");

    let gen = || Strategy::new_random(&board, MOVES);
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
