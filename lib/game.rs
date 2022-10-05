//! Basic implementation of the free daily puzzle
//! [figure.game](https://figure.game).

use std::{
    fmt,
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
    GameError : {
        OffBoard => "location is off the board",
        OffBoardMulti => "encountered off-board location",
    }
);
pub type GameResult<T> = Result<T, GameError>;

mkerr!(
    IOError : {
        ReadError => "couldn't read config file",
        ParseError => "couldn't parse config file",
        KeyNotFound => "missing expected key(s)",
        TypeError => "couldn't coerce type",
    }
);
pub type IOResult<T> = Result<T, IOError>;

/// Block type, including the case where one has been removed.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Block {
    White = 1,
    Yellow = 2,
    Green = 3,
    Purple = 4,
    Empty = 0,
}

impl From<Block> for String {
    fn from(block: Block) -> String {
        return match block {
            Block::Empty => "( )",
            Block::White => "(W)",
            Block::Yellow => "(Y)",
            Block::Green => "(G)",
            Block::Purple => "(P)",
        }.to_string();
    }
}

impl From<Block> for u8 {
    fn from(block: Block) -> u8 {
        return block as u8;
    }
}

impl From<u8> for Block {
    fn from(u: u8) -> Block {
        return match u {
            1 => Block::White,
            2 => Block::Yellow,
            3 => Block::Green,
            4 => Block::Purple,
            _ => Block::Empty,
        };
    }
}

impl fmt::Display for Block {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        return write!(f, "{}", String::from(*self));
    }
}

/// Rectangular grid of `Block`s. Indices are arranged such that `(0, 0)` is the
/// bottom-left corner of the board.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Board {
    blocks: Vec<Block>,
    pub shape: (usize, usize),
}

impl Board {
    /// Generate a new, empty board.
    pub fn new(shape: (usize, usize)) -> Self {
        return Board {
            blocks: vec![Block::Empty; shape.0 * shape.1],
            shape,
        };
    }

    /// Generate a new board filled with randomly chosen blocks.
    pub fn new_random(shape: (usize, usize)) -> Self {
        let mut rng: rnd::ThreadRng = rnd::thread_rng();
        let blocks: Vec<Block>
            = (0..shape.0 * shape.1)
            .map(|_| rng.gen_range(1..5).into())
            .collect();
        return Board { blocks, shape };
    }

    /// Collect an iterator of blocks into a shape. Takes only the first `N`
    /// blocks that would fit in the board if the iterator is too long or pads
    /// out the end with `Block::Empty` if the iterator is too short.
    pub fn from_iter_shape<I, T>(iter: I, shape: (usize, usize)) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<Block>,
    {
        let mut blocks: Vec<Block>
            = iter.into_iter().take(shape.0 * shape.1)
            .map(|b| b.into()).collect();
        if blocks.len() < shape.0 * shape.1 {
            blocks.append(
                &mut vec![Block::Empty; shape.0 * shape.1 - blocks.len()]
            );
        }
        return Board { blocks, shape };
    }

    /// Set the block type at a certain index, overwriting the previous value.
    pub fn set_block(&mut self, idx: (usize, usize), block: Block)
        -> GameResult<()>
    {
        return self.get_mut(idx.0, idx.1)
            .map(|b| { *b = block; })
            .ok_or(GameError::OffBoard);
    }

    /// Set block types at multiple indices, overwriting the previous values.
    pub fn set_block_multi<I>(&mut self, blocks: I) -> GameResult<()>
    where I: IntoIterator<Item = ((usize, usize), Block)>
    {
        for (idx, b) in blocks.into_iter() {
            self.set_block(idx, b)?;
        }
        return Ok(());
    }

    /// Return `true` if all blocks are `Block::Empty`, `false` otherwise.
    pub fn is_empty(&self) -> bool {
        return self.blocks.iter().map(|b| u8::from(*b)).sum::<u8>() == 0;
    }

    /// Return the number of non-empty blocks remaining on the board.
    pub fn non_empty(&self) -> usize {
        return self.blocks.iter()
            .map(|b| if b == &Block::Empty { 0 } else { 1 })
            .sum::<usize>();
    }

    /// Return an iterator over all blocks on the board, including empty ones.
    /// Blocks are visited in column-first order. The iterator type is `&Block`.
    pub fn iter(&self) -> BoardIter { self.blocks.iter() }

    /// Return an iterator over all blocks with associated indices on the board,
    /// including empty ones. Blocks are visited in column-first order. The 
    /// iterator type is `((usize, usize), &Block)`.
    pub fn indexed_iter(&self) -> BoardIndexedIter {
        return BoardIndexedIter { board: self, cur: 0, len: self.blocks.len() };
    }

    /// Convert a single number `k` to an index pair `(m, n)` if it is valid for
    /// the board.
    pub fn ravel(&self, k: usize) -> Option<(usize, usize)> {
        return if k < self.blocks.len() {
            Some((k / self.shape.1, k.rem_euclid(self.shape.1)))
        } else {
            None
        };
    }

    /// Convert an index pair `(m, n)` to a single number `k` if it is valid for
    /// the board.
    pub fn unravel(&self, m: usize, n: usize) -> Option<usize> {
        return if m < self.shape.0 && n < self.shape.1 {
            Some(m * self.shape.1 + n)
        } else {
            None
        };
    }

    /// Indices start with `(0, 0)` at the bottom left corner of the board.
    pub fn get(&self, row: usize, col: usize) -> Option<&Block> {
        return self.unravel(row, col)
            .map(|k| self.blocks.get(k).unwrap());
    }

    /// Indices start with `(0, 0)` at the bottom left corner of the board.
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Block> {
        return self.unravel(row, col)
            .map(|k| self.blocks.get_mut(k).unwrap());
    }

    /// Return references to the neighbors of the given location with their
    /// indices.
    pub fn get_neighbors(&self, row: usize, col: usize)
        -> Vec<((usize, usize), &Block)>
    {
        let mut neighbors: Vec<((usize, usize), &Block)> = Vec::new();
        let mut check: Vec<(usize, usize)> = Vec::with_capacity(4);
        if row < self.shape.0 { check.push((row + 1, col)) }
        if row > 0            { check.push((row - 1, col)) }
        if col < self.shape.1 { check.push((row, col + 1)) }
        if col > 0            { check.push((row, col - 1)) }
        for (m, n) in check.into_iter() {
            if let Some(b) = self.get(m, n) {
                neighbors.push(((m, n), b));
            } else {
                continue;
            }
        }
        return neighbors;
    }

    /// Get the indices of all the blocks connected with the same color to the
    /// block at `(row, col)`. Returns an empty sequence if the target block is
    /// empty.
    pub fn get_group(&self, row: usize, col: usize)
        -> GameResult<Vec<(usize, usize)>>
    {
        let target: &Block = self.get(row, col).ok_or(GameError::OffBoard)?;
        let mut idx: Vec<(usize, usize)> = vec![(row, col)];
        let mut checked: Vec<(usize, usize)> = vec![(row, col)];
        let mut frontier: Vec<((usize, usize), &Block)>
            = self.get_neighbors(row, col);
        while let Some(((m, n), b)) = frontier.pop() {
            if !checked.contains(&(m, n)) && b == target {
                idx.push((m, n));
                checked.push((m, n));
                frontier.append(&mut self.get_neighbors(m, n));
            } else {
                continue;
            }
        }
        return Ok(idx);
    }

    /// Block locations are checked before any removals are performed.
    pub fn remove_blocks<I>(&mut self, idx: I) -> GameResult<()>
    where I: IntoIterator<Item = (usize, usize)>
    {
        let to_remove: Vec<(usize, usize)> = idx.into_iter().collect();
        if to_remove.iter().any(|(m, n)| self.get(*m, *n).is_none()) {
            return Err(GameError::OffBoardMulti);
        }
        for (m, n) in to_remove.into_iter() {
            *self.get_mut(m, n).unwrap() = Block::Empty;
        }
        return Ok(());
    }

    /// Shift all blocks down such that every column has no empty blocks below
    /// any non-empty block.
    pub fn shift_down(&mut self) {
        let mut non_empty_above: bool;
        let mut m: isize;
        for n in 0..self.shape.1 {
            non_empty_above = false;
            m = self.shape.0 as isize - 1;
            while m >= 0 {
                if self.get(m as usize, n).unwrap() != &Block::Empty {
                    non_empty_above = true;
                } else if non_empty_above {
                    *self.get_mut(m as usize, n).unwrap()
                        = *self.get(m as usize + 1, n).unwrap();
                    *self.get_mut(m as usize + 1, n).unwrap() = Block::Empty;
                    non_empty_above = false;
                    m = self.shape.0 as isize - 1;
                    continue;
                }
                m -= 1;
            }
        }
    }

    /// Perform a move; i.e. select a group starting from the bottom row at
    /// `(0, k)`, remove it, and shift everything down.
    pub fn do_move(&mut self, k: usize) -> GameResult<()> {
        let group: Vec<(usize, usize)> = self.get_group(0, k)?;
        self.remove_blocks(group)?;
        self.shift_down();
        return Ok(());
    }

    /// Perform multiple moves.
    pub fn do_move_multi<I>(&mut self, moves: I) -> GameResult<()>
    where I: IntoIterator<Item = usize>
    {
        for k in moves.into_iter() {
            self.do_move(k)?;
        }
        return Ok(());
    }
}

pub type BoardIter<'a> = std::slice::Iter<'a, Block>;

pub struct BoardIndexedIter<'a> {
    board: &'a Board,
    cur: usize,
    len: usize,
}

impl<'a> Iterator for BoardIndexedIter<'a> {
    type Item = ((usize, usize), &'a Block);

    fn next(&mut self) -> Option<Self::Item> {
        let ret: Option<Self::Item>;
        if self.cur < self.len {
            let idx: (usize, usize)
                = (
                    self.cur / self.board.shape.1,
                    self.cur.rem_euclid(self.board.shape.1)
                );
            ret = Some((idx, self.board.blocks.get(self.cur).unwrap()));
            self.cur += 1;
        } else {
            ret = None;
        }
        return ret;
    }
}

impl IntoIterator for Board {
    type Item = Block;

    type IntoIter = std::vec::IntoIter<Block>;

    fn into_iter(self) -> Self::IntoIter { self.blocks.into_iter() }
}

impl<'a> IntoIterator for &'a Board {
    type Item = &'a Block;

    type IntoIter = std::slice::Iter<'a, Block>;

    fn into_iter(self) -> Self::IntoIter { self.blocks.iter() }
}

impl<const M: usize, const N: usize, T> From<[[T; N]; M]> for Board
where T: Into<Block>
{
    fn from(grid: [[T; N]; M]) -> Self {
        let blocks: Vec<Block>
            = grid.into_iter()
            .flat_map(|row| row.into_iter())
            .map(|b| b.into())
            .collect();
        return Board { blocks, shape: (M, N) };
    }
}

impl fmt::Display for Board {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut outstr = String::new();
        let mut rowstr = "[ ".to_string();
        for (k, b) in self.blocks.iter().enumerate() {
            rowstr += &(String::from(*b) + " ");
            if k.rem_euclid(self.shape.1) == self.shape.1 - 1 {
                outstr = (rowstr + "]\n") + &outstr;
                // outstr += &(rowstr + "]\n");
                rowstr = "[ ".to_string();
            }
        }
        return write!(f, "{}", outstr.trim());
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    pub blocks: Vec<u8>,
    pub n_moves: usize,
}

impl Config {
    pub fn from_file(infile: PathBuf) -> IOResult<Self> {
        let table: toml::Value
            = fs::read_to_string(infile.clone())
            .map_err(|_| IOError::ReadError)?
            .parse::<toml::Value>()
            .map_err(|_| IOError::ParseError)?;
        let blocks: Vec<u8>;
        if let Some(X) = table.get("blocks") {
            let _blocks_: Vec<i64>
                = X.clone().try_into().map_err(|_| IOError::TypeError)?;
            blocks = _blocks_.into_iter().map(|b| b as u8).collect();
        } else {
            return Err(IOError::KeyNotFound);
        }
        let n_moves: usize;
        if let Some(X) = table.get("n_moves") {
            let _n_moves_: i64
                = X.clone().try_into().map_err(|_| IOError::TypeError)?;
            n_moves = _n_moves_ as usize;
        } else {
            return Err(IOError::KeyNotFound);
        }
        return Ok(Config { blocks, n_moves });
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn board_group() -> GameResult<()> {
        let into_blocks: [[u8; 5]; 5] = [
            [ 1, 3, 2, 3, 1 ],
            [ 4, 2, 3, 1, 1 ],
            [ 4, 1, 3, 1, 1 ],
            [ 3, 2, 4, 2, 2 ],
            [ 2, 1, 1, 2, 2 ],
        ];
        let board = Board::from(into_blocks);
        let group: Vec<(usize, usize)> = board.get_group(0, 4)?;
        assert!(
            [(0, 4), (1, 4), (1, 3), (2, 4), (2, 3)]
                .into_iter().all(|idx| group.contains(&idx))
        );
        assert!(group.len() == 5);
        return Ok(());
    }

    #[test]
    fn board_remove_group() -> GameResult<()> {
        let mut init = Board::from([
            [ 1, 3, 2, 3, 1 ],
            [ 4, 2, 3, 1, 1 ],
            [ 4, 1, 3, 1, 1 ],
            [ 3, 2, 4, 2, 2 ],
            [ 2, 1, 1, 2, 2 ],
        ]);
        let group: Vec<(usize, usize)> = init.get_group(0, 4)?;
        init.remove_blocks(group)?;

        let expected = Board::from([
            [ 1, 3, 2, 3, 0 ],
            [ 4, 2, 3, 0, 0 ],
            [ 4, 1, 3, 0, 0 ],
            [ 3, 2, 4, 2, 2 ],
            [ 2, 1, 1, 2, 2 ],
        ]);
        assert!(&init == &expected);
        assert!(!init.is_empty());
        return Ok(());
    }

    #[test]
    fn do_move() -> GameResult<()> {
        let mut init = Board::from([
            [ 1, 3, 2, 3, 1 ],
            [ 4, 2, 3, 1, 1 ],
            [ 4, 1, 3, 1, 1 ],
            [ 3, 2, 4, 2, 2 ],
            [ 2, 1, 1, 2, 2 ],
        ]);
        init.do_move(4)?;

        let expected = Board::from([
            [ 1, 3, 2, 3, 2 ],
            [ 4, 2, 3, 2, 2 ],
            [ 4, 1, 3, 2, 0 ],
            [ 3, 2, 4, 0, 0 ],
            [ 2, 1, 1, 0, 0 ],
        ]);
        assert!(&init == &expected);
        return Ok(());
    }

    #[test]
    fn do_move_multi() -> GameResult<()> {
        let mut init = Board::from([
            [ 1, 3, 2, 3, 1 ],
            [ 4, 2, 3, 1, 1 ],
            [ 4, 1, 3, 1, 1 ],
            [ 3, 2, 4, 2, 2 ],
            [ 2, 1, 1, 2, 2 ],
        ]);
        init.do_move_multi([4, 4, 0, 0, 4])?;

        let expected = Board::from([
            [ 3, 3, 2, 3, 0 ],
            [ 2, 2, 3, 0, 0 ],
            [ 0, 1, 3, 0, 0 ],
            [ 0, 2, 4, 0, 0 ],
            [ 0, 1, 1, 0, 0 ],
        ]);
        assert!(&init == &expected);
        assert!(init.non_empty() == 13);
        return Ok(());
    }

    #[test]
    fn clean_board() -> GameResult<()> {
        let mut init = Board::from([
            [ 1, 3, 2, 3, 1 ],
            [ 4, 2, 3, 1, 1 ],
            [ 4, 1, 3, 1, 1 ],
            [ 3, 2, 4, 2, 2 ],
            [ 2, 1, 1, 2, 2 ],
        ]);
        init.do_move_multi([0, 0, 2, 2, 4, 3, 3, 0, 1, 2])?;
        assert!(init.non_empty() == 0);
        assert!(init.is_empty());
        return Ok(());
    }
}

