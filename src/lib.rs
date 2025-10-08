use libc;

// Include the generated GLPK bindings
#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(improper_ctypes)]
#[allow(dead_code)]
mod glpk {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use libc::c_int;
use std::collections::HashMap;
use std::fmt;

pub type Bound = (i32, i32);
pub type ID<'a> = &'a str;
pub type Objective<'a> = HashMap<ID<'a>, f64>;
pub type Interpretation<'a> = HashMap<ID<'a>, i64>;

/// Thin wrapper around the numeric GLPK constants that glpk-sys
/// (or your bindgen output) does not re-export.
///
/// Feel free to extend this file if you need more flags later on.
pub mod glp_consts {
    // Row/column bound types
    pub const GLP_FR:  i32 = 1;
    pub const GLP_LO:  i32 = 2;
    pub const GLP_UP:  i32 = 3;
    pub const GLP_DB:  i32 = 4;
    pub const GLP_FX:  i32 = 5;

    // Variable kinds
    pub const GLP_CV:  i32 = 1;
    pub const GLP_IV:  i32 = 2;
    pub const GLP_BV:  i32 = 3;

    // Optimisation direction
    pub const GLP_MIN: i32 = 1;
    pub const GLP_MAX: i32 = 2;

    // Solution / MIP status
    pub const GLP_UNDEF:  i32 = 1;
    pub const GLP_FEAS:   i32 = 2;
    pub const GLP_INFEAS: i32 = 3;
    pub const GLP_NOFEAS: i32 = 4;
    pub const GLP_OPT:    i32 = 5;
    pub const GLP_UNBND:  i32 = 6;
}

#[derive(Debug, PartialEq)]
pub enum Status {
    Undefined = 1,
    Feasible = 2,
    Infeasible = 3,
    NoFeasible = 4,
    Optimal = 5,
    Unbounded = 6,
    SimplexFailed = 7,
    MIPFailed = 8,
    EmptySpace = 9,
}

#[derive(Clone, Hash)]
pub struct Variable<'a> {
    pub id: ID<'a>,
    pub bound: Bound,
}


#[derive(Clone, Hash)]
pub struct IntegerSparseMatrix {
    pub rows: Vec<i32>,
    pub cols: Vec<i32>,
    pub vals: Vec<i32>,
}

#[derive(Hash)]
pub struct SparseLEIntegerPolyhedron<'a> {
    pub A: IntegerSparseMatrix,
    pub b: Vec<Bound>,
    pub variables: Vec<Variable<'a>>,
    pub double_bound: bool,
}

impl<'a> SparseLEIntegerPolyhedron<'a> {
    pub fn get_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        let hash = hasher.finish();
        hash
    }
}

impl fmt::Display for SparseLEIntegerPolyhedron<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let space = 4;
        let nr_rows = self.A.rows.iter().max().unwrap_or(&0) + 1;
        let nr_cols = self.A.cols.iter().max().unwrap_or(&0) + 1;
        // Create 2d array of zeroes
        let mut matrix = vec![vec![0; nr_cols as usize]; nr_rows as usize];
        for ((r, c), v) in self.A.rows.iter().zip(self.A.cols.iter()).zip(self.A.vals.iter()) {
            matrix[*r as usize][*c as usize] = *v;
        }
        for variable in &self.variables {
            write!(f, "{:>space$}", &variable.id.chars().take(3).collect::<String>())?;
        }
        writeln!(f)?;
        for (idx, row) in matrix.iter().enumerate() {
            if self.double_bound {
                write!(f, "{:>space$} <= ", self.b[idx].0)?;
            }
            for val in row.iter() {
                write!(f, "{:>space$}", val)?;
            }
            write!(f, " <= {:>space$}", self.b[idx].1)?;
            writeln!(f)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Solution {
    pub status: Status,
    pub objective: i32,
    pub solution: HashMap<String, i64>,
    pub error: Option<String>,
}

pub struct SolutionResponse {
    pub solutions: Vec<Solution>,
}

// 1    GLP_FR      Free variable (−∞ < x < ∞) 
// 2    GLP_LO      Variable with lower bound (x ≥ l) 
// 3    GLP_UP      Variable with upper bound (x ≤ u) 
// 4    GLP_DB      Double-bounded variable (l ≤ x ≤ u) 
// 5    GLP_FX      Fixed variable (x = l = u) 

// Value	Symbol	Meaning	Example
// 1	GLP_FR	Free (no bounds)	−∞<x<∞
// 2	GLP_LO	Lower bound only	x≥l
// 3	GLP_UP	Upper bound only	𝑥≤𝑢
// 4	GLP_DB	Double bounded (both bounds)	l≤x≤u
// 5	GLP_FX	Fixed (both bounds are equal)	x=l=u

// Variable kind
// 1 - continuous, 2 - integer, 3 - binary

// Solution status
// GLP_UNDEF   1  /* Solution is undefined */
// GLP_FEAS    2  /* Solution is feasible */
// GLP_INFEAS  3  /* No feasible solution exists */
// GLP_NOFEAS  4  /* No feasible solution exists (dual) */
// GLP_OPT     5  /* Optimal solution found */
// GLP_UNBND   6  /* Problem is unbounded */

/// Solves a set of integer linear programming (ILP) problems defined by the given polytope and objectives.
/// The function ignores objective variables not present in the polytope. All other variables not provided in objective but present in polytope will be set to 0.
///
/// # Arguments
///
/// * `polytope` - A reference to a `Polytope` struct that defines the constraints and variables of the ILP.
/// * `objectives` - A vector of `Objective` structs that define the objective functions to be optimized.
/// * `term_out` - A boolean flag indicating whether to enable terminal output for the GLPK solver.
///
/// # Returns
///
/// A vector of `Solution` structs, each representing the result of solving the ILP for one of the objectives.
///
/// # Panics
///
/// This function will panic if:
/// * The lengths of the rows, columns, and values in the constraint matrix `A` are not equal.
/// * The number of variables in the polytope does not match the maximum column index in the constraint matrix.
///
/// # Examples
///
/// ```
/// use glpk_rust::*;
/// use std::collections::HashMap;
/// 
/// let variables = vec![
///     Variable { id: "x1", bound: (0, 5) },
///     Variable { id: "x2", bound: (0, 3) },
/// ];
/// 
/// let mut polytope = SparseLEIntegerPolyhedron {
///     A: IntegerSparseMatrix {
///         rows: vec![0, 1],
///         cols: vec![0, 1], 
///         vals: vec![1, 1],
///     },
///     b: vec![(0, 10), (0, 8)],
///     variables,
///     double_bound: true,
/// };
/// 
/// let mut objective = HashMap::new();
/// objective.insert("x1", 1.0);
/// objective.insert("x2", 1.0);
/// let objectives = vec![objective];
/// 
/// let solutions = solve_ilps(&mut polytope, objectives, false, false);
/// for solution in solutions {
///     println!("{:?}", solution);
/// }
/// ```
pub fn solve_ilps<'a>(polytope: &mut SparseLEIntegerPolyhedron<'a>, objectives: Vec<Objective<'a>>, maximize: bool, term_out: bool) -> Vec<Solution> {

    // Initialize an empty vector to store solutions
    let mut solutions: Vec<Solution> = Vec::new();

    // Check that n_rows in shape is at least as large as the max row index in A
    let n_cols = polytope.variables.len();

    // If the polytope is empty, return an empty space status solution
    if polytope.A.rows.is_empty() || polytope.A.cols.is_empty() {
        panic!("The constraint matrix A cannot be empty, at the moment. This will be supported in future versions.");
    }

    // Get the maximum row and column indices from the constraint matrix A
    let poly_n_cols = (*polytope.A.cols.iter().max().unwrap() + 1) as usize;
    if n_cols < poly_n_cols {
        panic!("The number of variables must be at least as large as the maximum column index in the constraint matrix, got ({},{})", n_cols, poly_n_cols);
    }

    // Check if rows, columns and values are the same lenght. Else panic
    if (polytope.A.rows.len() != polytope.A.cols.len()) || (polytope.A.rows.len() != polytope.A.vals.len()) || (polytope.A.cols.len() != polytope.A.vals.len()) {
        panic!("Rows, columns and values must have the same length, got ({},{},{})", polytope.A.rows.len(), polytope.A.cols.len(), polytope.A.vals.len());
    }

    // Check that the number of rows is equal to the length of b
    if polytope.A.rows.iter().max().unwrap() + 1 > polytope.b.len() as i32 {
        panic!("The number of elements in b must be at least as large as the maximum row index in the constraint matrix, got ({},{})", polytope.b.len(), polytope.A.rows.iter().max().unwrap() + 1);
    }

    unsafe {

        // Enable or disable terminal output
        glpk::glp_term_out(term_out as c_int);

        // Create the problem
        let direction = if maximize { 2 } else { 1 };
        let lp = glpk::glp_create_prob();
        glpk::glp_set_obj_dir(lp, direction);

        // Add constraints (rows)
        glpk::glp_add_rows(lp, polytope.b.len() as i32);
        for (i, &b) in polytope.b.iter().enumerate() {
            let double_bound = if polytope.double_bound { glp_consts::GLP_DB } else { glp_consts::GLP_UP };
            glpk::glp_set_row_bnds(lp, (i + 1) as i32, double_bound, b.0 as f64, b.1 as f64);
        }

        // Add variables
        glpk::glp_add_cols(lp, polytope.variables.len() as i32);
        for (i, var) in polytope.variables.iter().enumerate() {
            
            // Set col bounds
            if var.bound.0 == var.bound.1 {
                glpk::glp_set_col_bnds(lp, (i + 1) as i32, glp_consts::GLP_FX, var.bound.0 as f64, var.bound.1 as f64);
            } else {
                glpk::glp_set_col_bnds(lp, (i + 1) as i32, glp_consts::GLP_DB, var.bound.0 as f64, var.bound.1 as f64);
            }

            // Set col kind - always integer variables
            // From GLPK docs: "Setting a column to GLP_BV has the same effect as if it were set to GLP_IV, its lower bound were
            // set 0, and its upper bound were set to 1."
            glpk::glp_set_col_kind(lp, (i + 1) as i32, glp_consts::GLP_IV);
        }

        // Set the constraint matrix
        let rows: Vec<i32> = std::iter::once(0).chain(polytope.A.rows.iter().map(|x| *x + 1)).collect();
        let cols: Vec<i32> = std::iter::once(0).chain(polytope.A.cols.iter().map(|x| *x + 1)).collect();
        let vals_f64: Vec<f64> = std::iter::once(0.0).chain(polytope.A.vals.iter().map(|x| *x as f64)).collect();

        // ne: The number of non-zero elements in the constraint matrix (not the number of rows or columns).
        // ia: An array of row indices (1-based) for each non-zero element.
        // ja: An array of column indices (1-based) for each non-zero element.
        // ar: An array of values corresponding to each non-zero element.
        glpk::glp_load_matrix(
            lp, 
            (vals_f64.len()-1) as i32, 
            rows.as_ptr(), 
            cols.as_ptr(), 
            vals_f64.as_ptr()
        );

        // Solve for multiple objectives
        for obj in objectives.iter() {

            // Setup empty solution
            let mut solution = Solution { status: Status::Undefined, objective: 0, solution: HashMap::new(), error: None };

            // Update the objective function
            for (j, v) in polytope.variables.iter().enumerate() {
                let coef = obj.get(&v.id).unwrap_or(&0.0);
                glpk::glp_set_obj_coef(lp, (j + 1) as i32, *coef as f64);
            }

            // Solve the integer problem using presolving
            let mut mip_params = glpk::glp_iocp::default();
            glpk::glp_init_iocp(&mut mip_params);
            mip_params.presolve = 1; 
            let mip_ret = glpk::glp_intopt(lp, &mip_params);

            if mip_ret != 0 {
                solution.status = Status::MIPFailed;
                solution.error = Some(format!("GLPK MIP solver failed with code: {}", mip_ret));
                solutions.push(solution);
                continue;
            }

            let status = glpk::glp_mip_status(lp);
            match status {
                1 => {
                    solution.status = Status::Undefined;
                    solution.error = Some("Solution is undefined".to_string());
                },
                2 => {
                    solution.status = Status::Feasible;
                    solution.objective = glpk::glp_mip_obj_val(lp) as i32;
                    for (j, var) in polytope.variables.iter().enumerate() {
                        let x = glpk::glp_mip_col_val(lp, (j + 1) as i32);
                        solution.solution.insert(var.id.to_string(), x as i64);
                    }
                },
                3 => {
                    solution.status = Status::Infeasible;
                    solution.error = Some("Infeasible solution exists".to_string());
                }
                4 => {
                    solution.status = Status::NoFeasible;
                    solution.error = Some("No feasible solution exists".to_string());
                }
                5 => {
                    solution.status = Status::Optimal;
                    solution.objective = glpk::glp_mip_obj_val(lp) as i32;
                    for (j, var) in polytope.variables.iter().enumerate() {
                        let x = glpk::glp_mip_col_val(lp, (j + 1) as i32);
                        solution.solution.insert(var.id.to_string(), x as i64);
                    }
                },
                6 => {
                    solution.status = Status::Unbounded;
                    solution.error = Some("Problem is unbounded".to_string());
                },
                x => {
                    panic!("Unknown status when solving ({})", x);
                }
            }
            solutions.push(solution);
        }

        // Clean up
        glpk::glp_delete_prob(lp);

        return solutions;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_enum() {
        assert_eq!(Status::Optimal as i32, 5);
        assert_eq!(Status::Infeasible as i32, 3);
    }

    #[test]
    fn test_variable_creation() {
        let var = Variable {
            id: "x1",
            bound: (0, 10),
        };
        assert_eq!(var.id, "x1");
        assert_eq!(var.bound, (0, 10));
    }

    #[test]
    fn test_sparse_le_polyhedron_creation() {
        let variables = vec![
            Variable { id: "x1", bound: (0, 5) },
            Variable { id: "x2", bound: (0, 3) },
        ];
        
        let polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 1],
                cols: vec![0, 1],
                vals: vec![2, 3]
    },
            b: vec![(0, 10), (0, 15)],
            variables,
            double_bound: true,
        };
        
        assert_eq!(polytope.A.vals, vec![2, 3]);
        assert_eq!(polytope.b, vec![(0, 10), (0, 15)]);
        assert!(polytope.double_bound);
        assert_eq!(polytope.variables.len(), 2);
    }

    #[test]
    fn test_simple_conjunction() {
        let variables = vec![
            Variable { id: "x1", bound: (0, 1) },
            Variable { id: "x2", bound: (0, 1) },
            Variable { id: "x3", bound: (0, 1) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 0, 0],
                cols: vec![0, 1, 2],
                vals: vec![1, 1, 1]
            },
            b: vec![(0, 3)],
            variables,
            double_bound: false,
        };
        let objectives = vec![HashMap::from_iter(vec![
            ("x1", 1.0),
            ("x2", 1.0),
            ("x3", 1.0),
        ])];
        let solutions = solve_ilps(&mut polytope, objectives, true, false);
        
        assert_eq!(solutions.len(), 1);
        let solution = &solutions[0];
        assert_eq!(solution.status, Status::Optimal);
        assert_eq!(solution.solution.get("x1"), Some(&1));
        assert_eq!(solution.solution.get("x2"), Some(&1));
        assert_eq!(solution.solution.get("x3"), Some(&1));
    }

    // This test should panic due to empty constraint matrix
    #[test]
    #[should_panic(expected = "The constraint matrix A cannot be empty, at the moment. This will be supported in future versions.")]
    fn test_empty_matrix_handling() {
        let variables = vec![];
        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![],
                cols: vec![],
                vals: vec![]
            },
            b: vec![],
            variables,
            double_bound: false,
        };
        
        let objectives = vec![HashMap::new()];
        let solutions = solve_ilps(&mut polytope, objectives, false, false);
        
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::EmptySpace);
    }

    #[test]
    fn test_single_variable_optimization() {
        let variables = vec![
            Variable { id: "x", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0],
                cols: vec![0],
                vals: vec![1]
            },
            b: vec![(0, 5)],
            variables,
            double_bound: true,
        };
        
        let mut objective = HashMap::new();
        objective.insert("x", 1.0);
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, false, false);
        
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);
        assert_eq!(solutions[0].solution.get("x"), Some(&0));
    }

    #[test]
    fn test_binary_variables() {
        let variables = vec![
            Variable { id: "b1", bound: (0, 1) },
            Variable { id: "b2", bound: (0, 1) },
        ];
        
        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 0],
                cols: vec![0, 1],
                vals: vec![1, 1]
            },
            b: vec![(1, 1)],
            variables,
            double_bound: false,
        };
        
        let mut objective = HashMap::new();
        objective.insert("b1", 1.0);
        objective.insert("b2", 1.0);
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, true, false);
        
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);
        let b1_val = solutions[0].solution.get("b1").unwrap();
        let b2_val = solutions[0].solution.get("b2").unwrap();
        assert_eq!(b1_val + b2_val, 1);
    }

    #[test]
    fn test_fixed_variables() {
        let variables = vec![
            Variable { id: "fixed", bound: (5, 5) },
            Variable { id: "free", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 0],
                cols: vec![0, 1],
                vals: vec![1, 1]
            },
            b: vec![(0, 15)],
            variables,
            double_bound: false,
        };
        
        let mut objective = HashMap::new();
        objective.insert("free", 1.0);
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, true, false);
        
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);
        assert_eq!(solutions[0].solution.get("fixed"), Some(&5));
        assert_eq!(solutions[0].solution.get("free"), Some(&10));
    }

    #[test]
    fn test_infeasible_problem() {
        let variables = vec![
            Variable { id: "x", bound: (5, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0],
                cols: vec![0],
                vals: vec![1]
            },
            b: vec![(0, 3)],
            variables,
            double_bound: true,
        };
        
        let mut objective = HashMap::new();
        objective.insert("x", 1.0);
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, false, false);
        
        assert_eq!(solutions.len(), 1);
        assert!(matches!(solutions[0].status, Status::MIPFailed));
    }

    #[test]
    fn test_maximize_vs_minimize() {
        let variables = vec![
            Variable { id: "x", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0],
                cols: vec![0],
                vals: vec![1]
            },
            b: vec![(0, 5)],
            variables,
            double_bound: true,
        };
        
        let mut objective = HashMap::new();
        objective.insert("x", 1.0);
        let objectives = vec![objective.clone()];
        
        // Test minimization
        let min_solutions = solve_ilps(&mut polytope, objectives.clone(), false, false);
        assert_eq!(min_solutions[0].solution.get("x"), Some(&0));
        
        // Test maximization
        let max_solutions = solve_ilps(&mut polytope, objectives, true, false);
        assert_eq!(max_solutions[0].solution.get("x"), Some(&5));
    }

    #[test]
    fn test_multiple_objectives() {
        let variables = vec![
            Variable { id: "x", bound: (0, 10) },
            Variable { id: "y", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 0],
                cols: vec![0, 1],
                vals: vec![1, 1]
            },
            b: vec![(0, 5)],
            variables,
            double_bound: true,
        };
        
        let mut obj1 = HashMap::new();
        obj1.insert("x", 1.0);
        
        let mut obj2 = HashMap::new();
        obj2.insert("y", 1.0);
        
        let objectives = vec![obj1, obj2];
        
        let solutions = solve_ilps(&mut polytope, objectives, false, false);
        
        assert_eq!(solutions.len(), 2);
        assert_eq!(solutions[0].status, Status::Optimal);
        assert_eq!(solutions[1].status, Status::Optimal);
    }

    #[test]
    fn test_objective_with_missing_variables() {
        let variables = vec![
            Variable { id: "x", bound: (0, 10) },
            Variable { id: "y", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 0],
                cols: vec![0, 1],
                vals: vec![1, 1]
            },
            b: vec![(0, 5)],
            variables,
            double_bound: true,
        };
        
        let mut objective = HashMap::new();
        objective.insert("x", 1.0);
        objective.insert("z", 5.0); // Variable not in polytope
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, false, false);
        
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);
        assert_eq!(solutions[0].solution.get("y"), Some(&0));
    }

    #[test]
    fn test_negative_coefficients() {
        let variables = vec![
            Variable { id: "x", bound: (0, 10) },
            Variable { id: "y", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 0],
                cols: vec![0, 1],
                vals: vec![-1, 1]
            },
            b: vec![(0, 5)],
            variables,
            double_bound: true,
        };
        
        let mut objective = HashMap::new();
        objective.insert("x", 1.0);
        objective.insert("y", 1.0);
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, false, false);
        
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);
    }

    #[test]
    fn test_zero_coefficient_matrix() {
        let variables = vec![
            Variable { id: "x", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0],
                cols: vec![0],
                vals: vec![0]
            },
            b: vec![(0, 5)],
            variables,
            double_bound: true,
        };
        
        let mut objective = HashMap::new();
        objective.insert("x", 1.0);
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, false, false);
        
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);
    }

    #[test]
    fn test_large_bounds() {
        let variables = vec![
            Variable { id: "x", bound: (0, 1000000) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0],
                cols: vec![0],
                vals: vec![1]
            },
            b: vec![(0, 500000)],
            variables,
            double_bound: true,
        };
        
        let mut objective = HashMap::new();
        objective.insert("x", 1.0);
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, true, false);
        
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);
        assert_eq!(solutions[0].solution.get("x"), Some(&500000));
    }

    #[test]
    fn test_json_data_polyhedron() {
        let variables = vec![
            Variable { id: "a", bound: (0, 1) },
            Variable { id: "b", bound: (0, 1) },
            Variable { id: "18a7bec7bbb9fe127d6107f77af0f11b24a6a846", bound: (0, 1) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 0, 0, 1, 1, 1, 2],
                cols: vec![0, 1, 2, 0, 1, 2, 2],
                vals: vec![-1, -1, 2, 1, 1, -2, -1]
    },
            b: vec![(0, 1), (0, 0), (0, -1)],
            variables,
            double_bound: false,
        };
        
        let mut objective = HashMap::new();
        objective.insert("a", 1.0);
        objective.insert("b", -1.0);
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, true, false);
        
        assert_eq!(solutions.len(), 1);
        let solution = &solutions[0];
        assert_eq!(solution.status, Status::Optimal);
        assert_eq!(solution.objective, 1);
        assert_eq!(solution.solution.get("a"), Some(&1));
        assert_eq!(solution.solution.get("b"), Some(&0));
    }

    #[test]
    fn test_complex_constraint_system() {
        let variables = vec![
            Variable { id: "x1", bound: (0, 10) },
            Variable { id: "x2", bound: (0, 10) },
            Variable { id: "x3", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 0, 0, 1, 1, 1],
                cols: vec![0, 1, 2, 0, 1, 2],
                vals: vec![1, 2, 3, 2, 1, 1]
            },
            b: vec![(0, 15), (0, 10)],
            variables,
            double_bound: true,
        };
        
        let mut objective = HashMap::new();
        objective.insert("x1", 3.0);
        objective.insert("x2", 2.0);
        objective.insert("x3", 1.0);
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, true, false);
        
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);
    }

    #[test]
    fn test_double_bound_vs_single_bound() {
        let variables = vec![
            Variable { id: "x", bound: (2, 8) },
        ];

        let mut polytope_double = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0],
                cols: vec![0],
                vals: vec![1]
            },
            b: vec![(0, 10)],
            variables: variables.clone(),
            double_bound: true,
        };

        let mut polytope_single = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0],
                cols: vec![0],
                vals: vec![1]
            },
            b: vec![(0, 10)],
            variables,
            double_bound: false,
        };
        
        let mut objective = HashMap::new();
        objective.insert("x", 1.0);
        let objectives = vec![objective];
        
        let solutions_double = solve_ilps(&mut polytope_double, objectives.clone(), false, false);
        let solutions_single = solve_ilps(&mut polytope_single, objectives, false, false);
        
        assert_eq!(solutions_double.len(), 1);
        assert_eq!(solutions_single.len(), 1);
        assert_eq!(solutions_double[0].status, Status::Optimal);
        assert_eq!(solutions_single[0].status, Status::Optimal);
    }

    #[test]
    fn test_sparse_matrix_with_gaps() {
        let variables = vec![
            Variable { id: "x1", bound: (0, 10) },
            Variable { id: "x2", bound: (0, 10) },
            Variable { id: "x3", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 2],
                cols: vec![0, 2],
                vals: vec![1, 1]
            },
            b: vec![(0, 5), (0, 5), (0, 5)],
            variables,
            double_bound: true,
        };
        
        let mut objective = HashMap::new();
        objective.insert("x1", 1.0);
        objective.insert("x2", 1.0);
        objective.insert("x3", 1.0);
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, false, false);
        
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);
    }

    #[test]
    #[should_panic(expected = "Rows, columns and values must have the same length")]
    fn test_mismatched_matrix_dimensions() {
        let variables = vec![
            Variable { id: "x", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 1],
                cols: vec![0],
                vals: vec![1]
            },
            b: vec![(0, 5)],
            variables,
            double_bound: true,
        };
        
        let objectives = vec![HashMap::new()];
        solve_ilps(&mut polytope, objectives, false, false);
    }

    #[test]
    #[should_panic(expected = "The number of variables must be at least as large as the maximum column index in the constraint matrix, got (1,2)")]
    fn test_variable_column_mismatch() {
        let variables = vec![
            Variable { id: "x", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0],
                cols: vec![1],
                vals: vec![1]
            },
            b: vec![(0, 5)],
            variables,
            double_bound: true,
        };
        
        let objectives = vec![HashMap::new()];
        solve_ilps(&mut polytope, objectives, false, false);
    }

    #[test]
    #[should_panic(expected = "The number of elements in b must be at least as large as the maximum row index in the constraint matrix, got (1,2)")]
    fn test_row_bound_mismatch() {
        let variables = vec![
            Variable { id: "x", bound: (0, 10) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 1],
                cols: vec![0, 0],
                vals: vec![1, 1]
            },
            b: vec![(0, 5)],
            variables,
            double_bound: true,
        };
        
        let objectives = vec![HashMap::new()];
        solve_ilps(&mut polytope, objectives, false, false);
    }

    #[test]
    fn test_fixed_boolean_variables() {
        // Test that boolean variables can be fixed to specific values
        // Using bounds that won't be classified as binary to avoid GLPK binary variable behavior
        let variables = vec![
            Variable { id: "fixed_zero", bound: (0, 0) },
            Variable { id: "fixed_one", bound: (1, 1) },
            Variable { id: "free_binary", bound: (0, 1) },
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 0, 0],
                cols: vec![0, 1, 2],
                vals: vec![1, 1, 1]
    },
            b: vec![(0, 10)],
            variables,
            double_bound: false,
        };
        
        let mut objective = HashMap::new();
        objective.insert("fixed_zero", 1.0);
        objective.insert("fixed_one", 1.0);
        objective.insert("free_binary", 1.0);
        let objectives = vec![objective];
        
        let solutions = solve_ilps(&mut polytope, objectives, true, false);
        
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);
        
        // Fixed to 0 should remain 0 even when classified as binary
        assert_eq!(solutions[0].solution.get("fixed_zero"), Some(&0));
        
        // Fixed to 1 should remain 1 even when classified as binary
        assert_eq!(solutions[0].solution.get("fixed_one"), Some(&1));
        
        // Free binary variable should be able to take value 0 or 1
        // With maximization, it should be 1
        assert_eq!(solutions[0].solution.get("free_binary"), Some(&1));
    }

    #[test]
    fn test_atmost_one_of_constraints_with_bounds_fixed_in_variables() {
        // 0 <= [  1,   1,   1,   3] <=   4
        // 0 <= [ -1,  -1,  -1,  -5] <=  -2
        let variables = vec![
            Variable { id: "a", bound: (0, 1) },
            Variable { id: "b", bound: (0, 1) },
            Variable { id: "c", bound: (0, 1) },
            Variable { id: "A", bound: (1, 1) } // Fixed to 1 means at most one of a,b,c can be 1
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 0, 0, 0, 1, 1, 1, 1],
                cols: vec![3, 0, 1, 2, 3, 0, 1, 2],
                vals: vec![3, 1, 1, 1, -5, -1, -1, -1]
    },
            b: vec![(0, 4), (0, -2)],
            variables,
            double_bound: false,
        };

        // Try to take a b and c
        let objective = HashMap::from([("b", 1.0), ("a", 1.0), ("c", 1.0)]);
        let objectives = vec![objective];
        let solutions = solve_ilps(&mut polytope, objectives, true, false);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);

        // Check that at most one of a,b,c is 1
        let sum_abc = solutions[0].solution.get("a").unwrap() + solutions[0].solution.get("b").unwrap() + solutions[0].solution.get("c").unwrap();
        assert!(sum_abc <= 1);
    }

    #[test]
    fn test_that_shape_too_small() {
        let variables = vec![
            Variable { id: "a", bound: (0, 1) },
            Variable { id: "b", bound: (0, 1) },
            Variable { id: "c", bound: (0, 1) },
            Variable { id: "R", bound: (0, 1) } 
        ];

        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0],
                cols: vec![0],
                vals: vec![1]
    },
            b: vec![(0, 1)],
            variables,
            double_bound: false,
        };
        let objective = HashMap::from([("a", 1.0), ("b", 1.0), ("c", 1.0), ("R", 1.0)]);
        let objectives = vec![objective];
        let solutions = solve_ilps(&mut polytope, objectives, true, false);
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].status, Status::Optimal);
        // Sum solution and check that it is 4
        let sum: i64 = solutions[0].solution.values().sum();
        assert_eq!(sum, 4);
    }
}
