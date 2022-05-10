import swiglpk
import cplex
from mmodel import *

def make_cplex_prob(model):
    problem = cplex.Cplex()
    problem.variables.add(
        names=model.rxns,
        ub=model.ub,
        lb=model.lb,)
    
    S = model.S.tocsc()
    
    lin_expr = []
    senses = []
    rhs = []
    names = []
    for i in range(len(model.mets)):
        lin_expr.append(cplex.SparsePair(S[:,[i]].indices.tolist(), S[:,[i]].data.tolist()))
        senses.append('E')
        rhs.append(0)
        names.append(model.mets[i])

    problem.linear_constraints.add(
        lin_expr=lin_expr,
        senses=senses,
        rhs=rhs,
        names=names)
    
    problem.objective.set_linear(zip(model.rxns, model.get_penalties_vector(0)))
    problem.objective.set_name('reaction_penalties')
    problem.objective.set_sense(problem.objective.sense.minimize)
    
    return problem

def cplex_reset_obj(problem):
    problem.objective.set_linear(zip(problem.variables.get_names(), [0] * len(problem.variables.get_names())))
    problem.objective.set_name('none')
    

class Cplex_Solver:
    def __init__(self, problem, model, solver_fn):
        self.problem = problem
        self.model = model
        self.solver_fn = solver_fn
        
    def solve(self, rxn_indices, sample_indices):
        times = []
        res = []
        prev_sample_ind = -1
        cache = None
        for i in range(len(rxn_indices)):
            rxn_ind = rxn_indices[i]
            sample_ind = sample_indices[i]
            
            if sample_ind != prev_sample_ind:
                self.problem.objective.set_linear(zip(self.model.rxns, self.model.get_penalties_vector(sample_ind)))
            
            #Assuming unidirectional for now
            rxn = self.model.rxns[rxn_ind]
            r_max = self.model.cache[rxn]
            
            #High flux constraint
            self.problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[rxn], val=[1.0])], 
                                               senses=['E'], rhs=[BETA*r_max], names=['REACTION_OPT'])
            
            #Block reverse reaction constraint (only needed in unidirectional)
            partner_rxn = self.model.partner_rxns.get(rxn, None)
            if partner_rxn:
                self.problem.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=[partner_rxn], val=[1.0])], 
                                                   senses=['E'], rhs=[0], names=['BLOCK_REV'])
                    
            start_time = perf_counter()
            obj_val, cache = self.solver_fn(self.problem, cache)
            res.append(obj_val)
            
            times.append(perf_counter() - start_time)
            
            self.problem.linear_constraints.delete(['REACTION_OPT'])
            
            if partner_rxn:
                self.problem.linear_constraints.delete(['BLOCK_REV'])
                
        return res, times
    
    @staticmethod
    def simple_solver_fn(problem, cache):
        problem.solve()
        return problem.solution.get_objective_value(), None
    

def make_glpk_problem(model):
    lp = swiglpk.glp_create_prob()
    swiglpk.glp_set_obj_dir(lp, swiglpk.GLP_MIN)
    
    n_rows = len(model.mets)
    swiglpk.glp_add_rows(lp, n_rows)
    for i in range(n_rows):
        swiglpk.glp_set_row_name(lp, i+1, model.mets[i])
        swiglpk.glp_set_row_bnds(lp, i+1, swiglpk.GLP_FX, 0.0, 0.0)
        
    n_cols = len(model.rxns)
    swiglpk.glp_add_cols(lp, n_cols)
    for i in range(n_cols):
        swiglpk.glp_set_col_name(lp, i+1, model.rxns[i])
        swiglpk.glp_set_col_bnds(lp, i+1, swiglpk.GLP_DB, model.lb[i], model.ub[i])
        swiglpk.glp_set_obj_coef(lp, i+1, model.penalties[i,0])
    
    #Can't pass in numpy arrays from what I can tell, so just
    N = model.S.getnnz()
    A = swiglpk.intArray(N)
    B = swiglpk.intArray(N)
    C = swiglpk.doubleArray(N)
    for i in range(N):
        A[i+1] = int(model.S.col[i]) + 1
        B[i+1] = int(model.S.row[i]) + 1
        C[i+1] = model.S.data[i]
    swiglpk.glp_load_matrix(lp, N, A, B, C)
    return lp


class Glpk_Solver:
    def __init__(self, problem, model, solver_fn):
        self.problem = problem
        self.model = model
        self.solver_fn = solver_fn
        
    def solve(self, rxn_indices, sample_indices):
        times = []
        res = []
        prev_sample_ind = -1
        cache = None
        for i in range(len(rxn_indices)):
            rxn_ind = int(rxn_indices[i])
            sample_ind = int(sample_indices[i])
            
            if sample_ind != prev_sample_ind:
                penalty_vector = self.model.get_penalties_vector(sample_ind)
                for i in range(swiglpk.glp_get_num_cols(self.problem)):
                    swiglpk.glp_set_obj_coef(self.problem, i+1, penalty_vector[i])
            
            #Assuming unidirectional for now
            rxn = self.model.rxns[rxn_ind]
            r_max = self.model.cache[rxn]
            
            old_lb, old_ub, old_type = (swiglpk.glp_get_col_lb(self.problem, rxn_ind+1), 
                                        swiglpk.glp_get_col_ub(self.problem, rxn_ind+1), 
                                        swiglpk.glp_get_col_type(self.problem, rxn_ind+1))
            
            #High flux constraint
            swiglpk.glp_set_col_bnds(self.problem, rxn_ind+1, swiglpk.GLP_FX, BETA*r_max, BETA*r_max)
            
            #Block reverse reaction constraint (only needed in unidirectional)
            partner_rxn = self.model.partner_rxns.get(rxn, None)
            if partner_rxn:
                partner_rxn_ind = self.model.rxns.index(partner_rxn)
                p_old_lb, p_old_ub, p_old_type = (swiglpk.glp_get_col_lb(self.problem, partner_rxn_ind+1), 
                                                  swiglpk.glp_get_col_ub(self.problem, partner_rxn_ind+1),
                                                  swiglpk.glp_get_col_type(self.problem, partner_rxn_ind+1))
                
                swiglpk.glp_set_col_bnds(self.problem, partner_rxn_ind+1, swiglpk.GLP_FX, BETA*r_max, BETA*r_max)
                    
            start_time = perf_counter()
            obj_val, cache = self.solver_fn(self.problem, cache)
            res.append(obj_val)
            times.append(perf_counter() - start_time)
            
            swiglpk.glp_set_col_bnds(self.problem, rxn_ind+1, old_type, old_lb, old_ub)
            
            if partner_rxn:
                swiglpk.glp_set_col_bnds(self.problem, partner_rxn_ind+1, p_old_type, p_old_lb, p_old_ub)
                
        return res, times
    
    @staticmethod
    def basic_simplex_solver_fn(lp, cache):
        swiglpk.glp_simplex(lp, None)
        return swiglpk.glp_get_obj_val(lp), None