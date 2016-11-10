
#include "smurfs_mf.hpp"


void mf_solve_binary_mrf(const int* edges, const float* unary_cost,
						 const float* pairwise_cost, const float* label_cost,
						 int *result,
						 int n_nodes, int n_edges)
{
	Graph<float, float, float> *G;
	G = new Graph<float,float,float>(n_nodes, n_edges);

	int n_labels = 2;
	int ushift, eshift;
	float e00, e01, e10, e11;
	int e1, e2;
	float pw;

	int i, e;
	float *uptr, *pptr;
	int *eptr;

	G->add_node(n_nodes);

	for ( i = 0, uptr = unary_cost; i < n_nodes;  i++, uptr += n_labels ) {
		G->add_tweights(i, *(uptr), *(uptr+1));
	}


	for ( e = 0, eptr = edges, pptr = pairwise_cost;
		  e < n_edges; e++, eptr += 2, pptr++ )
	{
		e1 = *eptr;
		e2 = *(eptr+1);
		pw = (*pptr) * label_cost[1];
		G->add_edge(e1, e2, pw, pw);
	}

	G->maxflow();

	for ( i = 0; i < n_nodes; i++ ) {
		result[i] = G->what_segment(i);
	}

	delete G;
}


void mf_solve_mrf(const int* edges, const float* unary_cost,
				  const float* pairwise_cost, const float* label_cost,
				  int *result,
				  int n_nodes, int n_edges, int n_labels, int n_iter,
				  unsigned long random_seed)
{
	srand((unsigned)42);
	Graph<float, float, float> *G;
	G = new Graph<float,float,float>(n_nodes, n_edges);

	int i, e;
	float *uptr, *pptr;
	int *eptr;
	float e00, e01, e10, e11;
	int e1, e2;
	float pw, pwd, pw1, pw2;

	int changes, alpha, dummy, label;
	int min_changes = (int)(0.1f * n_nodes);

	for ( int n = 0; n < n_iter; n++ )
	{
		changes = 0;

		for ( int alpha = 0; alpha < n_labels; alpha++ ) {

			G->reset();
			G->add_node(n_nodes);

			for ( i = 0, uptr = unary_cost; i < n_nodes;  i++, uptr += n_labels ) {
				if ( result[i] == alpha ) {
					G->add_tweights(i, *(uptr + alpha), FLT_MAX);
				} else {
					G->add_tweights(i, *(uptr + alpha), *(uptr + result[i]));
				}
			}

			for ( e = 0, eptr = edges, pptr = pairwise_cost;
				  e < n_edges; e++, eptr += 2, pptr++ )
			{
				e1 = *(eptr);
				e2 = *(eptr + 1);
				pw = *(pptr);

				if ( result[e1] == result[e2] ) {
					pw *= label_cost[result[e1] * n_labels + alpha];
					G->add_edge(e1, e2, pw, pw);
				} else {
					pwd = pw * label_cost[result[e1] * n_labels + result[e2]];
					pw1 = pw * label_cost[result[e1] * n_labels + alpha];
					pw2 = pw * label_cost[result[e2] * n_labels + alpha];
					dummy = G->add_node(1);
					G->add_tweights(dummy, 0, pwd);
					G->add_edge(e1, dummy, pw1, pw1);
					G->add_edge(e2, dummy, pw2, pw2);
				}
			}

			G->maxflow();

			for ( i = 0; i < n_nodes; i++ ) {
				label = G->what_segment(i, G->SOURCE);
				if ( label == G->SINK && result[i] != alpha ) {
					result[i] = alpha;
					changes += 1;
				}
			}
		}

		if ( changes < min_changes ) {
			break;
		}
	}

	delete G;
}


void mf_solve_binary_smurfs(const int* edges, const float* unary_cost,
							const float* pairwise_cost, const float* label_cost,
							int *result,
							int n_nodes, int n_edges,
							const int* regions, int n_regions)
{
	Graph<float, float, float> *G;
	G = new Graph<float,float,float>(n_nodes, n_edges);

	int n_labels = 2;
	int ushift, eshift;
	float e00, e01, e10, e11;
	int e1, e2;
	float pw;

	int i, e;
	float *uptr, *pptr;
	int *eptr;

	G->add_node(n_nodes);

	for ( i = 0, uptr = unary_cost; i < n_nodes;  i++, uptr += n_labels ) {
		G->add_tweights(i, *uptr, *(uptr+1));
	}


	for ( e = 0, eptr = edges, pptr = pairwise_cost;
		  e < n_edges; e++, eptr += 2, pptr++ )
	{
		e1 = *eptr;
		e2 = *(eptr+1);
		pw = (*pptr) * label_cost[regions[e1] * 2 + 1];
		G->add_edge(e1, e2, pw, pw);
	}

	G->maxflow();

	for ( i = 0; i < n_nodes; i++ ) {
		result[i] = G->what_segment(i);
	}

	delete G;
}


void mf_solve_smurfs(const int* edges, const float* unary_cost,
					 const float* pairwise_cost, const float* label_cost,
					 int *result,
					 int n_nodes, int n_edges, int n_labels, int n_iter,
					 const int *regions, int n_region,
					 unsigned long random_seed)
{
	std::srand( unsigned(random_seed) );
	Graph<float, float, float> *G;
	G = new Graph<float,float,float>(n_nodes, n_edges);

	int i, e;
	float *uptr, *pptr;
	int *eptr;
	float e00, e01, e10, e11;
	int e1, e2, rshift;
	float pw, pwd, pw1, pw2;

	int changes, alpha, dummy, label;
	int min_changes = (int)(0.1f * n_nodes);

	for ( int n = 0; n < n_iter; n++ )
	{
		changes = 0;

		for ( int alpha = 0; alpha < n_labels; alpha++ ) {
			G->reset();
			G->add_node(n_nodes);

			for ( i = 0, uptr = unary_cost; i < n_nodes;  i++, uptr += n_labels ) {
				if ( result[i] == alpha ) {
					G->add_tweights(i, *(uptr + alpha), FLT_MAX);
				} else {
					G->add_tweights(i, *(uptr + alpha), *(uptr + result[i]));
				}
			}

			for ( e = 0, eptr = edges, pptr = pairwise_cost;
				  e < n_edges; e++, eptr += 2, pptr++ )
			{
				e1 = *(eptr);
				e2 = *(eptr + 1);
				pw = *(pptr);
				rshift = regions[e1] * n_labels;

				if ( result[e1] == result[e2] ) {
					pw *= label_cost[rshift + result[e1] * n_labels + alpha];
					G->add_edge(e1, e2, pw, pw);
				} else {
					pwd = pw * label_cost[rshift + result[e1] * n_labels + result[e2]];
					pw1 = pw * label_cost[rshift + result[e1] * n_labels + alpha];
					pw2 = pw * label_cost[rshift + result[e2] * n_labels + alpha];
					dummy = G->add_node(1);
					G->add_tweights(dummy, 0, pwd);
					G->add_edge(e1, dummy, pw1, pw1);
					G->add_edge(e2, dummy, pw2, pw2);
				}
			}

			G->maxflow();

			for ( i = 0; i < n_nodes; i++ ) {
				label = G->what_segment(i, G->SOURCE);
				if ( label == G->SINK && result[i] != alpha ) {
					result[i] = alpha;
					changes += 1;
				}
			}

			if ( changes < min_changes ) {
				break;
			}
		}
	}

	delete G;
}
