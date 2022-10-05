//! Drivers for the genetic algorithm.

use std::{
    fmt,
};
use rand::{
    prelude as rnd,
    Rng,
};
use crate::{
    mkerr,
    game::Board,
    evolve::{
        Probability,
        Score,
        Propagate,
        Population,
    },
};

mkerr!(
    SolveError : {
        InvalidStrategy => "strategy couldn't be applied to its board",
        UnequalBoards => "strategies must be for the same board",
        BadProbability => "probability must be between 0 and 1",
    }
);
pub type SolveResult<T> = Result<T, SolveError>;

impl Score for isize { }

/// Holds a sequence of moves encoded as the position of the block whose group
/// is to be removed, starting with 0 at the left-most column.
///
/// Each `Strategy` carries a reference to an associated `Board` in an initial
/// state, and has fitness equal to the negative of the number of blocks
/// remaining on the board after the moves have been performed. The board is
/// cloned in order to evaluate the fitness of a strategy, and the value is then
/// cached.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Strategy<'a> {
    board: &'a Board,
    moves: Vec<usize>,
    score: Option<isize>,
}

impl<'a> Strategy<'a> {
    /// Create a new strategy for a board.
    pub fn new(board: &'a Board, moves: Vec<usize>) -> Self { 
        return Strategy { board, moves, score: None };
    }

    pub fn new_random_rng<R>(board: &'a Board, N_moves: usize, rng: &mut R)
        -> Self
    where R: Rng + ?Sized
    {
        let moves: Vec<usize>
            = (0..N_moves).map(|_| rng.gen_range(0..board.shape.1)).collect();
        return Strategy { board, moves, score: None };
    }

    /// Generate a new strategy with moves uniformly sampled from all possible.
    pub fn new_random(board: &'a Board, N_moves: usize) -> Self {
        let mut rng: rnd::ThreadRng = rnd::thread_rng();
        return Strategy::new_random_rng(board, N_moves, &mut rng);
    }

    pub fn get_board(&self) -> &'a Board { self.board }

    pub fn get_moves(&self) -> &Vec<usize> { &self.moves }

    pub fn get_moves_mut(&mut self) -> &mut Vec<usize> { &mut self.moves }

    pub fn iter(&self) -> std::slice::Iter<usize> {
        return self.moves.iter();
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<usize> {
        return self.moves.iter_mut();
    }

    /// Evaluate the fitness of the strategy if it has not yet been evaluated.
    pub fn eval_score(&mut self) -> SolveResult<()> {
        if self.score.is_none() {
            let mut board = self.board.clone();
            board.do_move_multi(self.moves.iter().copied())
                .map_err(|_| SolveError::InvalidStrategy)?;
            self.score = Some(-(board.non_empty() as isize));
        }
        return Ok(());
    }

    /// Get the fitness score if it has been evaluated.
    pub fn get_score(&self) -> Option<isize> { self.score }

    /// Evaluate the fitness of the strategy if it has not yet been evaluated
    /// and return the result.
    pub fn score(&mut self) -> SolveResult<isize> {
        self.eval_score()?;
        return Ok(self.score.unwrap());
    }

    /// Return a new `Board` in its final state after applying the strategy.
    pub fn final_board(&self) -> SolveResult<Board> {
        let mut board = self.board.clone();
        board.do_move_multi(self.moves.iter().copied())
            .map_err(|_| SolveError::InvalidStrategy)?;
        return Ok(board);
    }
}

impl<'a> Propagate for Strategy<'a> {
    type Fitness = isize;

    type Error = SolveError;

    fn crossover_rng<R>(&self, other: &Self, rng: &mut R)
        -> SolveResult<Self>
    where R: Rng + ?Sized
    {
        if self.get_board() != other.get_board() {
            return Err(SolveError::UnequalBoards);
        }
        let moves: Vec<usize>
            = self.iter().zip(other.iter())
            .map(|(l, r)| if rng.gen::<bool>() { *l } else { *r })
            .collect();
        return Ok(Strategy::new(self.get_board(), moves));
    }

    /// Generate a new strategy for the same board by taking moves element-wise
    /// from two parent strategies with equal probability. Fails if the parent
    /// strategies are for different boards.
    fn crossover(&self, other: &Self) -> SolveResult<Self> {
        let mut rng: rnd::ThreadRng = rnd::thread_rng();
        return self.crossover_rng(other, &mut rng);
    }

    fn mutate_rng<R>(&mut self, p: Probability, rng: &mut R)
    where R: Rng + ?Sized
    {
        let move_max: usize = self.get_board().shape.1;
        for m in self.get_moves_mut().iter_mut() {
            if Probability::random_rng(rng).ok_unchecked(p.p()) {
                *m = rng.gen_range(0..move_max);
            }
        }
        self.score = None;
    }
    
    /// Mutate the moves in the strategy individually, each with probability
    /// `p`. New moves are chosen from the space of possible moves for the board
    /// with uniform probability.
    fn mutate(&mut self, p: Probability) {
        let mut rng: rnd::ThreadRng = rnd::thread_rng();
        return self.mutate_rng(p, &mut rng);
    }

    /// Evaluate the fitness of the strategy by performing it on (a cloned
    /// instance of) its board and returning the number of blocks remaining as
    /// the fitness metric.
    fn eval(&mut self) -> SolveResult<isize> {
        return self.score();
   }
}

impl fmt::Display for Strategy<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(f, "Strategy{:?}", self.moves);
    }
}

/// Driver for the genetic algorithm. Provides convenience methods for acting on
/// all individuals.
///
/// Should be instantiated through `Population::init`.
#[derive(Clone, Debug)]
pub struct Solver<'a> {
    strategies: Vec<Strategy<'a>>,
}

impl<'a> Solver<'a> {
    pub fn get_strategies(&self) -> &Vec<Strategy> { &self.strategies }

    /// Evaluate the fitness of all individuals.
    pub fn eval(&mut self) -> SolveResult<()> {
        for s in self.strategies.iter_mut() {
            s.eval()?;
        }
        return Ok(());
    }

    pub fn mutate_rng<R>(&mut self, mutate: Probability, rng: &mut R)
    where R: Rng + ?Sized
    {
        for s in self.strategies.iter_mut() {
            s.mutate_rng(mutate, rng);
        }
    }

    /// Mutate all individuals.
    pub fn mutate(&mut self, mutate: Probability) {
        let mut rng: rnd::ThreadRng = rnd::thread_rng();
        self.mutate_rng(mutate, &mut rng);
    }

    /// Sort individuals in decreasing fitness order.
    pub fn sort(&mut self) {
        self.strategies.sort_by(|l, r| {
            l.get_score().unwrap().cmp(&r.get_score().unwrap())
        });
        self.strategies.reverse()
    }
}

impl<'a> Population for Solver<'a> {
    type Fitness = isize;

    type Error = SolveError;

    type Individual = Strategy<'a>;

    fn init<F>(N: usize, gen: F) -> Self
    where F: Fn() -> Strategy<'a>
    {
        let strategies: Vec<Strategy> = (0..N).map(|_| gen()).collect();
        return Solver { strategies };
    }

    fn evolve_rng<R>(&mut self, mutate: Probability, rng: &mut R)
        -> SolveResult<()>
    where R: Rng
    {
        let N: usize = self.strategies.len();
        self.mutate_rng(mutate, rng);
        self.eval()?; // fitness evaluation is cached
        self.sort();
        let (mut a, mut b): (usize, usize);
        for k in N / 2 + 1..N {
            a = rng.gen_range(0..N / 2 + 1);
            b = rng.gen_range(0..N / 2 + 1);
            while b == a {
                b = rng.gen_range(0..N / 2 + 1);
            }
            self.strategies[k]
                = self.strategies[a].crossover_rng(&self.strategies[b], rng)?;
        }
        self.eval()?;
        return Ok(());
    }

    fn get_most_fit(&self) -> Option<&Strategy<'a>> {
        let mut ret: Option<&Strategy<'a>> = None;
        for s in self.strategies.iter() {
            if let Some(b) = s.get_score() {
                if let Some(Some(a)) = ret.map(|s0| s0.get_score()) {
                    if b > a {
                        ret = Some(s);
                    }
                } else {
                    ret = Some(s);
                }
            } else {
                continue;
            }
        }
        return ret;
    }
}


