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

pub type Bound = (i32, i32);
pub type ID = String;
pub type Objective = HashMap<ID, f64>;
pub type Interpretation = HashMap<ID, i64>;

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

#[derive(Clone)]
pub struct Variable {
    pub id: ID,
    pub bound: Bound,
}

#[derive(Clone)]
pub struct Shape {
    pub nrows: usize,
    pub ncols: usize,
}

#[derive(Clone)]
pub struct IntegerSparseMatrix {
    pub rows: Vec<i32>,
    pub cols: Vec<i32>,
    pub vals: Vec<i32>,
    pub shape: Shape,
}

pub struct SparseLEIntegerPolyhedron {
    pub A: IntegerSparseMatrix,
    pub b: Vec<Bound>,
    pub variables: Vec<Variable>,
    pub double_bound: bool,
}

#[derive(Debug)]
pub struct Solution {
    pub status: Status,
    pub objective: i32,
    pub solution: Interpretation,
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
///     Variable { id: "x1".to_string(), bound: (0, 5) },
///     Variable { id: "x2".to_string(), bound: (0, 3) },
/// ];
/// 
/// let mut polytope = SparseLEIntegerPolyhedron {
///     A: IntegerSparseMatrix {
///         rows: vec![0, 1],
///         cols: vec![0, 1], 
///         vals: vec![1, 1],
///         shape: Shape { nrows: 2, ncols: 2 },
///     },
///     b: vec![(0, 10), (0, 8)],
///     variables,
///     double_bound: true,
/// };
/// 
/// let mut objective = HashMap::new();
/// objective.insert("x1".to_string(), 1.0);
/// objective.insert("x2".to_string(), 1.0);
/// let objectives = vec![objective];
/// 
/// let solutions = solve_ilps(&mut polytope, objectives, false, false);
/// for solution in solutions {
///     println!("{:?}", solution);
/// }
/// ```
pub fn solve_ilps(polytope: &mut SparseLEIntegerPolyhedron, objectives: Vec<Objective>, maximize: bool, term_out: bool) -> Vec<Solution> {

    // Initialize an empty vector to store solutions
    let mut solutions: Vec<Solution> = Vec::new();

    // Check if rows, columns and values are the same lenght. Else panic
    if (polytope.A.rows.len() != polytope.A.cols.len()) || (polytope.A.rows.len() != polytope.A.vals.len()) || (polytope.A.cols.len() != polytope.A.vals.len()) {
        panic!("Rows, columns and values must have the same length, got ({},{},{})", polytope.A.rows.len(), polytope.A.cols.len(), polytope.A.vals.len());
    }

    // If the polytope is empty, return an empty space status solution
    if polytope.A.rows.is_empty() || polytope.A.cols.is_empty() {
        for _ in 0..objectives.len() {
            solutions.push(Solution { status: Status::EmptySpace, objective: 0, solution: Interpretation::new(), error: None });
        }
        return solutions;
    }

    // Check that the max number of columns is equal to the number of provided variables
    let n_cols = (*polytope.A.cols.iter().max().unwrap() + 1) as usize;
    if polytope.variables.len() != n_cols {
        panic!("The number of variables must be equal to the maximum column index in the constraint matrix, got ({},{})", polytope.variables.len(), n_cols);
    }

    let n_rows = (*polytope.A.rows.iter().max().unwrap() + 1) as usize;
    if n_rows != polytope.b.len() {
        panic!("The number of rows in the constraint matrix must be equal to the number of elements in the b vector, got ({},{})", n_rows, polytope.b.len());
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
            let double_bound = if polytope.double_bound { 4 } else { 3 };
            glpk::glp_set_row_bnds(lp, (i + 1) as i32, double_bound, b.0 as f64, b.1 as f64);
        }

        // Add variables
        glpk::glp_add_cols(lp, polytope.variables.len() as i32);
        for (i, var) in polytope.variables.iter().enumerate() {
            
            // Set col bounds
            if var.bound.0 == var.bound.1 {
                glpk::glp_set_col_bnds(lp, (i + 1) as i32, 5, var.bound.0 as f64, var.bound.1 as f64);
            } else {
                glpk::glp_set_col_bnds(lp, (i + 1) as i32, 4, var.bound.0 as f64, var.bound.1 as f64);
            }

            // Set col kind - either integer (3) or binary (2) in this case
            if var.bound.0 == 0 && var.bound.1 == 1 {
                glpk::glp_set_col_kind(lp, (i + 1) as i32, 2);
            } else {
                glpk::glp_set_col_kind(lp, (i + 1) as i32, 3);
            };
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
            let mut solution = Solution { status: Status::Undefined, objective: 0, solution: Interpretation::new(), error: None };

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
                        solution.solution.insert(var.id.clone(), x as i64);
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
                        solution.solution.insert(var.id.clone(), x as i64);
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
            id: "x1".to_string(),
            bound: (0, 10),
        };
        assert_eq!(var.id, "x1");
        assert_eq!(var.bound, (0, 10));
    }

    #[test]
    fn test_sparse_le_polyhedron_creation() {
        let variables = vec![
            Variable { id: "x1".to_string(), bound: (0, 5) },
            Variable { id: "x2".to_string(), bound: (0, 3) },
        ];
        
        let polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![0, 1],
                cols: vec![0, 1],
                vals: vec![2, 3],
                shape: Shape { nrows: 2, ncols: 2 },
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
    fn test_empty_matrix_handling() {
        let variables = vec![];
        let mut polytope = SparseLEIntegerPolyhedron {
            A: IntegerSparseMatrix {
                rows: vec![],
                cols: vec![],
                vals: vec![],
                shape: Shape { nrows: 0, ncols: 0 },
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
}
