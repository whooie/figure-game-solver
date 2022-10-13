//! Machinery to facilitate the construction and evolution of a population of
//! individuals through a Darwinian process based on the evaluation of a local
//! fitness function.

use std::{
    convert::{
        TryFrom,
        TryInto,
    },
    fs,
    path::PathBuf,
};
use rand::{
    prelude as rnd,
    Rng,
};
use toml;
use crate::{
    mkerr,
};

mkerr!(
    EvolveError : {
        BadProbability => "probability must be between 0 and 1",
    }
);
pub type EvolveResult<T> = Result<T, EvolveError>;

mkerr!(
    IOError : {
        ReadError => "couldn't read config file",
        ParseError => "couldn't parse config file",
        TableNotFound => "missing expected table",
        KeyNotFound => "missing expected key(s)",
        TypeError => "couldn't coerce type",
        BadProbability => "probability must be between 0 and 1",
    }
);
pub type IOResult<T> = Result<T, IOError>;

/// Basic way to filter for floating point values in the range `[0.0, 1.0]`.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Probability {
    p: f64
}

impl Probability {
    /// Fails if `p` lies outside `[0.0, 1.0]`.
    pub fn new<P>(p: P) -> EvolveResult<Self>
    where P: TryInto<Probability, Error = EvolveError>
    {
        return p.try_into();
    }

    pub fn random_rng<R>(rng: &mut R) -> Self
    where R: Rng + ?Sized
    {
        return Probability { p: rng.gen::<f64>() };
    }

    /// Generate a random probability drawn uniformly from `[0.0, 1.0]`.
    pub fn random() -> Self {
        let mut rng: rnd::ThreadRng = rnd::thread_rng();
        return Probability { p: rng.gen::<f64>() };
    }

    /// Get the bare floating point value.
    pub fn p(&self) -> f64 { self.p }

    /// Compare `self` to another probability. Fails if `p` lies outside
    /// `[0.0, 1.0]`.
    pub fn ok<P>(&self, p: P) -> EvolveResult<bool>
    where P: TryInto<Probability, Error = EvolveError>
    {
        let threshold: Probability = p.try_into()?;
        return Ok(self.ok_unchecked(threshold.p()));
    }

    /// Compare `self` to a bare floating point value.
    pub fn ok_unchecked(&self, threshold: f64) -> bool { self.p <= threshold }
}

impl TryFrom<f64> for Probability {
    type Error = EvolveError;

    fn try_from(p: f64) -> EvolveResult<Probability> {
        return if (0.0..=1.0).contains(&p) {
            Ok(Probability { p })
        } else {
            Err(EvolveError::BadProbability)
        };
    }
}

impl TryFrom<f32> for Probability {
    type Error = EvolveError;

    fn try_from(p: f32) -> EvolveResult<Probability> {
        return if (0.0..=1.0).contains(&p) {
            Ok(Probability { p: p as f64 })
        } else {
            Err(EvolveError::BadProbability)
        };
    }
}

/// Basic sub-trait for quantifying individuals' fitnesses.
///
/// It is expected that greater values of types implementing this trait
/// correspond to greater fitnesses.
pub trait Score: PartialEq + PartialOrd { }

/// Describes the basic actions needed for evolution.
///
/// `eval` requires `self` to be mutable in order to allow for cached
/// evaluation.
pub trait Propagate
where Self: Sized
{
    type Fitness: Score;

    type Error;

    fn crossover_rng<R>(&self, other: &Self, rng: &mut R)
        -> Result<Self, Self::Error>
    where R: Rng + ?Sized;

    /// Generate a new individual by randomly selecting aspects from two
    /// parent individuals.
    fn crossover(&self, other: &Self) -> Result<Self, Self::Error> {
        let mut rng: rnd::ThreadRng = rnd::thread_rng();
        return self.crossover_rng(other, &mut rng);
    }

    fn mutate_rng<R>(&mut self, p: Probability, rng: &mut R)
    where R: Rng + ?Sized;

    /// Mutate aspects of an individual with some probability `p`.
    fn mutate(&mut self, p: Probability) {
        let mut rng: rnd::ThreadRng = rnd::thread_rng();
        return self.mutate_rng(p, &mut rng);
    }

    /// Evaluate the fitness of an individual.
    fn eval(&mut self) -> Result<Self::Fitness, Self::Error>;
}

/// Describes macro-level actions needed to evolve a population of propagators.
pub trait Population
{
    type Fitness: Score;

    type Error;

    type Individual: Propagate<Fitness = Self::Fitness, Error = Self::Error>;

    /// Basis constructor to initialize the population through repeated calls
    /// of `gen`.
    fn init<F>(N: usize, gen: F) -> Self
    where F: Fn() -> Self::Individual;

    fn evolve_rng<R>(&mut self, mutate: Probability, rng: &mut R)
        -> Result<(), Self::Error>
    where R: Rng;

    /// Evolve the population through a single generation.
    fn evolve(&mut self, mutate: Probability) -> Result<(), Self::Error> {
        let mut rng: rnd::ThreadRng = rnd::thread_rng();
        self.evolve_rng(mutate, &mut rng)?;
        return Ok(());
    }

    /// Evolve the population for a fixed number of generations.
    fn evolve_for(&mut self, mutate: Probability, n: usize)
        -> Result<(), Self::Error>
    {
        let mut rng: rnd::ThreadRng = rnd::thread_rng();
        for _ in 0..n {
            self.evolve_rng(mutate, &mut rng)?;
        }
        return Ok(());
    }

    /// Evolve the population until `cond` returns `true`.
    fn evolve_until<F>(&mut self, mutate: Probability, cond: F)
        -> Result<usize, Self::Error>
    where F: Fn(&Self) -> bool
    {
        let mut rng: rnd::ThreadRng = rnd::thread_rng();
        let mut k: usize = 0;
        loop {
            self.evolve_rng(mutate, &mut rng)?;
            k += 1;
            if cond(self) { break; }
        }
        return Ok(k);
    }

    /// Return a reference to the individual scoring the highest in fitness, if
    /// one can be determined.
    fn get_most_fit(&self) -> Option<&Self::Individual>;
}

#[derive(Clone, Debug)]
pub struct Config {
    pub n_pop: usize,
    pub mutate: Probability,
}

impl Config {
    pub fn from_file(infile: PathBuf) -> IOResult<Self> {
        let table: toml::Value
            = fs::read_to_string(infile)
            .map_err(|_| IOError::ReadError)?
            .parse::<toml::Value>()
            .map_err(|_| IOError::ParseError)?
            .get("evolve")
            .ok_or(IOError::TableNotFound)?
            .clone();
        let n_pop: usize;
        if let Some(X) = table.get("n_pop") {
            let _n_pop_: i64
                = X.clone().try_into().map_err(|_| IOError::TypeError)?;
            n_pop = _n_pop_ as usize;
        } else {
            return Err(IOError::KeyNotFound);
        }
        let mutate: Probability;
        if let Some(X) = table.get("mutate") {
            let _mutate_: f64
                = X.clone().try_into().map_err(|_| IOError::TypeError)?;
            mutate = _mutate_.try_into().map_err(|_| IOError::BadProbability)?;
        } else {
            return Err(IOError::KeyNotFound);
        }
        return Ok(Config { n_pop, mutate });
    }
}

