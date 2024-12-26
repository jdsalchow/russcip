use std::rc::Rc;
use russcip::*;

struct CSPPricer<'a> {
    stock_length: usize,
    item_sizes: &'a [f64],
    main_problem: ModelSolving,
}

impl Pricer for CSPPricer<'_> {
    fn generate_columns(&mut self, farkas: bool) -> PricerResult {
        let mut pricing_model = Model::new()
            .include_default_plugins()
            .create_prob("Pricing")
            .hide_output()
            .set_obj_sense(ObjSense::Maximize);

        let vars = (0..self.item_sizes.len()).map(|i| {
            let cons = self.main_problem.find_cons(&format!("demand_for_item_{i}")).unwrap();
            let dual_val = self.main_problem.dual_sol(cons);
            pricing_model.add_var(0.0,
                                  f64::INFINITY,
                                  dual_val,
                                  &format!("lambda_{i}"), VarType::Integer)
        }).collect::<Vec<Rc<Variable>>>();

        pricing_model.add_cons(vars.clone(), self.item_sizes, f64::NEG_INFINITY, self.stock_length as f64, "knapsack_constraint");
        let solved_model = pricing_model.solve();

        if solved_model.status() == Status::Optimal && !farkas && solved_model.best_sol().unwrap().obj_val() > 1.0 {
            let solution = solved_model.best_sol().unwrap();
            let pattern = vars.iter().map(|var| (solution.val(var.clone()) as u32).to_string()).collect::<Vec<String>>();

            // add variable for new cutting pattern
            let new_variable_name = &format!("pattern_{}", pattern.join(("-")));
            if self.main_problem.vars().iter().find(|var| &var.name() == new_variable_name).is_none() {
                let new_variable = self.main_problem.add_var(0.0, f64::INFINITY, 1.0, new_variable_name, VarType::Integer);

                (0..self.item_sizes.len()).for_each(|i| {
                    let constraint = self.main_problem.find_cons(&format!("demand_for_item_{i}")).unwrap();
                    self.main_problem.add_cons_coef(
                        constraint,
                        new_variable.clone(),
                        solution.val(vars[i].clone()),
                    );
                });

                PricerResult {
                    state: PricerResultState::FoundColumns,
                    lower_bound: Some(1.0 - solution.obj_val()),
                }
            } else {
                PricerResult {
                    state: PricerResultState::NoColumns,
                    lower_bound: None,
                }
            }
        } else {
            PricerResult {
                state: PricerResultState::NoColumns,
                lower_bound: None,
            }
        }
    }
}

// following https://scipbook.readthedocs.io/en/latest/bpp.html
fn main() {
    let stock_length = 9;
    let item_sizes = &[6.0, 5.0, 4.0, 2.0, 3.0, 7.0, 5.0, 8.0, 4.0, 5.0];
    let demand = &[2, 3, 4, 4, 2, 2, 2, 2, 2, 1];

    // Vector of cutting_patterns, initially populated with the trivial ones that contain exactly
    // one item. cutting_patterns[i][j] indicates how often item j is in pattern i.
    let initial_cutting_patterns: Vec<Vec<i32>> = (0..item_sizes.len()).map(|i|
        (0..item_sizes.len()).map(|j| if i == j { 1 } else { 0 }).collect::<Vec<i32>>()
    ).collect();

    let mut main_problem =
        Model::new()
            .include_default_plugins()
            .create_prob("Cutting Stock Problem")
            .set_presolving(ParamSetting::Off)
            .set_obj_sense(ObjSense::Minimize);

    let cutting_pattern_vars: Vec<Rc<Variable>> = initial_cutting_patterns.iter().enumerate().map(|(i, &ref _pattern)| {
        let pattern = (0..10).map(|x| if x == i { "1" } else { "0" }).collect::<Vec<_>>().join("-");
        main_problem.add_var(0.0, f64::INFINITY, 1.0, &format!("pattern_{pattern}"), VarType::Integer)
    }).collect();

    let demand_constraints = demand.iter().enumerate().map(|(i, &count)| {
        let cons = main_problem.add_cons(cutting_pattern_vars.clone(),
                                         &initial_cutting_patterns.iter().map(|pattern| pattern[i] as f64).collect::<Vec<f64>>(),
                                         count as f64,
                                         f64::INFINITY,
                                         &format!("demand_for_item_{i}"));
        main_problem.set_cons_modifiable(cons.clone(), true);
        cons
    }).collect::<Vec<Rc<Constraint>>>();
    println!("demand constraints{:?}", demand_constraints);

    let pricer = CSPPricer {
        stock_length,
        item_sizes,
        main_problem: main_problem.clone_for_plugins(),
    };

    let solved_model = main_problem
        .include_pricer("CSPPricer", "CSPPricer", 1, false, Box::new(pricer))
        .solve();

    let solution = solved_model.best_sol().unwrap();
    for var in solved_model.vars().iter() {
        let name = var.name();
        let value = solution.val(var.clone());
        if value != 0.0 {
            println!("{name}={value}")
        }
    }
}
