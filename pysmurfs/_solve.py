

from ..qpbo import solve_mrf, solve_smurfs, solve_binary_mrf, solve_binary_smurfs



def solve(edges, unary, pairwise, label_cost, niter=5):
	if unary.shape[1] > 2:
		if label_cost.ndim == 2:
			return solve_mrf(edges, unary, pairwise, label_cost, niter=5)
		else:
			return solve_smurfs(edges, unary, pairwise, label_cost, niter=5)
	else:
		if label_cost.ndim == 2:
			return solve_binary_mrf(edges, unary, pairwise, label_cost)
		else:
			return solve_binary_smurfs(edges, unary, pairwise, label_cost)
