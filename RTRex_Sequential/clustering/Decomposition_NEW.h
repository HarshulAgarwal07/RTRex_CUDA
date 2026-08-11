#ifndef ESCAPE_DECOMP_H_
#define ESCAPE_DECOMP_H_

#include "Escape/ErrorCode.h"
#include "Escape/Graph.h"
#include "Escape/ClusterStructures.h"
#include <stack>
#include <list>
#include <queue>
#include <algorithm>
#include <chrono>
#include <vector>
#include <map>
#include <iostream>
#include <inttypes.h>
#include <omp.h>

// all relevant structures are in ClusterStructures.h

using namespace Escape;
using namespace std;


// The common neighbor algorithm that produces the triangle weight
// Input: a pointer g to a CGraph that is sorted by ID (so binary search is possible).
// Output: a WeightedTriangleInfo for g. The ordering of edges in perEdge (of TriangleInfo) is that same as g.
// 
//   The algorithm does the common neighbor procedure for finding triangles. So it loops over every edge (u,v)
// and checks for common neighbors. The main trick is to search for the entries of the smaller list/degreee in
// the neighbor list of the other vertex. 

//   To ensure that every triangle is discovered exactly once, we use some tiebreaking rules. We process edge (u,v) only if
// u < v (in terms of id). Furthermore, we only consider the wedge (u,v,w) if w has the largest id. So triangle (u,v,w)
// is discovered exactly once: when we start with u, process edge (u,v), and intersect neighbor lists.
//
//   The weight of a triangle u, v, w is 1/{d(u)*d(v)*d(w)}, where d(u) is the degree of u. This is 
// based on defining the edge weight according to the normalized adjacency matrix (D^{-1/2}MD^{-1/2}).
//  


WeightedTriangleInfo commonNbr(CGraph *g)
{
   printf("Computing all triangle weighted info\n");

   WeightedTriangleInfo ret;   // output 
   // initialize output
   ret.total = 0;      
   ret.total_wgt = 0;  
   ret.perVertex = new Weight[g->nVertices+1];
   ret.perEdge = new Weight[g->nEdges+1]; 

   // initial all per vertex values to zero
   for (VertexIdx i=0; i < g->nVertices; ++i)
       ret.perVertex[i] = 0;

   // initial all per edge values to zero
   for (EdgeIdx j=0; j < g->nEdges; ++j)
       ret.perEdge[j] = 0; 

   // getting the partner map, so that we can update triangle weights more efficiently
   g->getPartnerMap(); // calling getPartnerMap, which returns an array

   // vertex variables
   VertexIdx v, w, lower, higher;
   EdgeIdx loc;
   Count degu, degv, degw; 
   Weight wgt;

   for (VertexIdx u=0; u < g->nVertices; ++u) // loop over vertices
       for (EdgeIdx i = g->offsets[u]; i < g->offsets[u+1]; ++i)   // loop over neighbor of u
       {
           v = g->nbors[i]; // get the neighbor v, present at location i in the nbors list

           degu = g->offsets[u+1] - g->offsets[u]; // get the degree of u
           degv = g->offsets[v+1] - g->offsets[v]; // get the degree of v

           // we will not process (u,v) if u > v. 
           if(u > v)
               continue;

           // we will now determine the neighbor with lower degree
           lower = (degu < degv) ? u : v; // checking if degu < degv
           higher = (degu < degv)? v : u; // getting the other vertex

           // loop of neighbors of lower, and look up these neighbors in the adj list for the other vertex
           for (EdgeIdx j = g->offsets[lower]; j < g->offsets[lower+1]; ++j)
           {
               w = g->nbors[j]; // get the neighbor w, preset at location j in the nbors list
               // we only consider this wedge if w has the largest id
               if (w <= u || w <= v)
                   continue;
               loc = g->getEdgeBinary(higher,w); // check if w is present in higher's nbors list
               if (loc != -1) // edge (upper, w) is present
               {
                   degw = g->offsets[w+1] - g->offsets[w]; // get the degree of w
                   wgt = 1/double(degu*degv*degw); // the weight of the triangle (u,v,w)
                   ret.total++; // increment the total number of triangles
                   ret.total_wgt += wgt; // add the weight to the total weight

                   ret.perEdge[i] += wgt; // add to the triangle weight of the edge (u,v), at location i
                   ret.perEdge[g->partnerMap[i]] += wgt; // edge (v,u), at the partner location for i
                   
                   ret.perEdge[j] += wgt; // (lower,w), at location j
                   ret.perEdge[g->partnerMap[j]] += wgt; // edge (w,lower), at the partner location for j

                   ret.perEdge[loc] += wgt; // (higher,w) at location loc
                   ret.perEdge[g->partnerMap[loc]] += wgt; // edge (w,higher), at the partner location for loc
                   
                   ret.perVertex[u] += wgt; // add to the triangle weight of vertex u
                   ret.perVertex[v] += wgt; // add to the triangle weight of vertex v
                   ret.perVertex[w] += wgt; // add to the triangle weight of vertex w
               }
           }
       }
    return ret;
}


/* deleteAndClean procedure: This is the real workhorse. It get a list of edges to delete (either from extraction or those that have already been detected
to be below the cleaning threshold). It deletes these edges, removes the corresponding triangles, and cleans all *adjacent* edges. Note that these are the
only edges whose triangle weights are affected. This process is propagated until convergence. Meaning, adjacent edges below the cleaning threshold
are deleted, which may lead to other adjacent edges being deleted, so on and so forth.

Input:
    g: A pointer to a CGraph
    edgeStatus: A pointer to a char array that stores the current status of the edges (Y = in the graph, N = not in the graph, S = in the stack). So the current graph has all 'N' edges already deleted.
    triInfo: A pointer to weighted triangle info structure, where weight are of the *current* graph. (So only triangles with non 'N' edges contribute to the weight.)
    toDelete: A pointer to a stack of edges to be deleted.
    eps: The cleaning threshold

Input assumption:
    - edgeStatus and toDelete are consistent. This means that edgeStatus[i] = 'S' iff the corresponding edge is in toDelete.
    - edgeStatus and triInfo are consistent. So triInfo only contains the weight of triangles whose edgeStatus is not 'N'

Output: void BUT the following inputs get changed
    The input assumptions will remain valid at the end of the procedure.
    - edgeStatus: edges will get deleted and be labeled 'N'. All edges in toDelete will get labeled 'N'.
    - triInfo: As edges get deleted, triInfo will be updated accordingly.
    - toDelete: This will become empty

The algorithm is as follows. We go over every edge in the stack. We list all the triangles that edge participates in. All the triangles
(with non 'N' edges) need to be deleted, so we reduce the triangle weight on all edges of the triangle. If any of these edges has
a triangle weight that goes below the cleaning threshold, we push it onto the slack toDelete. We keep continuing until the stack is empty.
*/

void deleteAndClean(CGraph *g, char *edgeStatus, WeightedTriangleInfo *triInfo, stack<Pair> *toDelete, double eps)
{
    // Process the toDelete stack to mark initial deletions
    while (!toDelete->empty())
    {
        Pair edge = toDelete->top(); // get the next edge to be deleted
        toDelete->pop();             // remove it from stack

        VertexIdx u = edge.first;
        VertexIdx v = edge.second;
        EdgeIdx loc_uv = g->getEdgeBinary(u, v);

        if (loc_uv != -1)
        {
            EdgeIdx partloc_uv = g->partnerMap[loc_uv];
            edgeStatus[loc_uv] = 'N';
            edgeStatus[partloc_uv] = 'N';
        }
    }

    long long V = g->nVertices;

    auto get_degree = [&](VertexIdx u)
    {
        return g->offsets[u + 1] - g->offsets[u];
    };

    auto is_directed = [&](VertexIdx u, VertexIdx v)
    {
        Count deg_u = get_degree(u);
        Count deg_v = get_degree(v);
        return deg_u < deg_v || (deg_u == deg_v && u < v);
    };

    vector<double> tri_weight(g->nEdges, 0.0);
    vector<double> vert_weight(V, 0.0);

    bool changed = true;
    long long total_tris = 0;
    double total_wt = 0.0;

    // ====================================================================
    // CASCADING DELETION LOOP
    // ====================================================================
    while (changed)
    {
        changed = false;

        fill(tri_weight.begin(), tri_weight.end(), 0.0);
        fill(vert_weight.begin(), vert_weight.end(), 0.0);
        total_tris = 0;
        total_wt = 0.0;

// 1. Count triangles using ONLY active edges
#pragma omp parallel
        {
            long long local_tris = 0;
            double local_wt = 0.0;

#pragma omp for schedule(dynamic)
            for (long long loop_u = 0; loop_u < V; ++loop_u)
            {
                VertexIdx u = loop_u;
                for (EdgeIdx e_uv = g->offsets[u]; e_uv < g->offsets[u + 1]; ++e_uv)
                {
                    if (edgeStatus[e_uv] == 'N')
                        continue;

                    VertexIdx v = g->nbors[e_uv];
                    if (!is_directed(u, v))
                        continue;

                    EdgeIdx i = g->offsets[u];
                    EdgeIdx j = g->offsets[v];
                    EdgeIdx end_u = g->offsets[u + 1];
                    EdgeIdx end_v = g->offsets[v + 1];

                    while (i < end_u && j < end_v)
                    {
                        VertexIdx w_u = g->nbors[i];
                        VertexIdx w_v = g->nbors[j];

                        if (w_u == w_v)
                        {
                            VertexIdx w = w_u;
                            if (edgeStatus[i] != 'N' && edgeStatus[j] != 'N')
                            {
                                if (is_directed(u, w) && is_directed(v, w))
                                {
                                    double d_u = get_degree(u);
                                    double d_v = get_degree(v);
                                    double d_w = get_degree(w);
                                    double wgt = 1.0 / (d_u * d_v * d_w);

#pragma omp atomic
                                    tri_weight[e_uv] += wgt;
#pragma omp atomic
                                    tri_weight[i] += wgt;
#pragma omp atomic
                                    tri_weight[j] += wgt;

#pragma omp atomic
                                    vert_weight[u] += wgt;
#pragma omp atomic
                                    vert_weight[v] += wgt;
#pragma omp atomic
                                    vert_weight[w] += wgt;

                                    local_tris++;
                                    local_wt += wgt;
                                }
                            }
                            i++;
                            j++;
                        }
                        else if (w_u < w_v)
                            i++;
                        else
                            j++;
                    }
                }
            }

#pragma omp atomic
            total_tris += local_tris;
#pragma omp atomic
            total_wt += local_wt;
        }

// 2. Symmetrize weights safely
#pragma omp parallel for schedule(dynamic)
        for (long long loop_u = 0; loop_u < V; ++loop_u)
        {
            VertexIdx u = loop_u;
            for (EdgeIdx e_uv = g->offsets[u]; e_uv < g->offsets[u + 1]; ++e_uv)
            {
                if (edgeStatus[e_uv] == 'N')
                    continue;

                VertexIdx v = g->nbors[e_uv];
                if (u < v)
                {
                    EdgeIdx e_vu = g->partnerMap[e_uv];
                    if (edgeStatus[e_vu] != 'N')
                    {
                        double total_wgt = tri_weight[e_uv] + tri_weight[e_vu];
                        tri_weight[e_uv] = total_wgt;
                        tri_weight[e_vu] = total_wgt;
                    }
                }
            }
        }

        // 3. Threshold Check
        bool local_changed = false;

#pragma omp parallel for reduction(| : local_changed)
        for (long long loop_u = 0; loop_u < V; ++loop_u)
        {
            VertexIdx u = loop_u;
            for (EdgeIdx e_uv = g->offsets[u]; e_uv < g->offsets[u + 1]; ++e_uv)
            {
                VertexIdx v = g->nbors[e_uv];

                if (u < v && edgeStatus[e_uv] != 'N')
                {
                    double d_u = get_degree(u);
                    double d_v = get_degree(v);
                    double edge_wgt = 1.0 / (d_u * d_v);

                    if (tri_weight[e_uv] < (eps * edge_wgt))
                    {
                        edgeStatus[e_uv] = 'N';

                        EdgeIdx e_vu = g->partnerMap[e_uv];
                        edgeStatus[e_vu] = 'N';

                        local_changed = true;
                    }
                }
            }
        }

        changed = local_changed;
    }

    // Assign final computed statistics to triInfo
    triInfo->total = total_tris;
    triInfo->total_wgt = total_wt;

    // We update the triInfo structures sequentially as there is no bottleneck here
    for (VertexIdx u = 0; u < V; ++u)
    {
        triInfo->perVertex[u] = vert_weight[u];
        for (EdgeIdx e_uv = g->offsets[u]; e_uv < g->offsets[u + 1]; ++e_uv)
        {
            triInfo->perEdge[e_uv] = tri_weight[e_uv];
        }
    }
}

/* initialClean procedure: it applies the repeated Jaccard based cleaning procedure. Every edge
with Jaccard similarity less than a parameter epsilon is removed from the graph. This process
is done iteratively until convergence. When the process ends, all remaining edges have
Jaccard value about epsilon.

Input:
    g: a pointer to a CGraph
    trInfo: a pointer to the corresponding WeightedTriangleInfo object
    eps: the cleaning threshold

Output:
    A char/byte array (with the same index as g->nbors) indicating which edges remain.
    A value of Y means edge is still present (not deleted), and N means edges was deleted during cleaning.
    We use a char array, since we will also store other flags. This is used in subsequent functions to mark intermediate steps of cleaning.

*/

char *initialClean(CGraph *g, WeightedTriangleInfo *triInfo, double eps)
{
    printf("Starting the initial clean\n");
    // set up the return value, which is a boolean array
    char *edgeStatus = new char[g->nEdges + 1];
    // no edge is deleted, so initialize all entries as Y
    for (EdgeIdx i = 0; i <= g->nEdges; i++)
        edgeStatus[i] = 'Y';

    stack<Pair> toDelete; // setting up a stack to store the edges that need to be deleted
    Count degu, degv;

    // loop over all the edges and find the ones whose Jaccard similarity is below the threshold eps
    // all these edges will be pushed onto the slack toDelete, and marked as "on the stack" ('S') in the array edgeStatus

    for (VertexIdx u = 0; u < g->nVertices; u++)
        for (EdgeIdx i = g->offsets[u]; i < g->offsets[u + 1]; ++i) // loop over neighbor of u
        {
            VertexIdx v = g->nbors[i]; // get the neighbor v, present at location i in the nbors list

            degu = g->offsets[u + 1] - g->offsets[u]; // get the degree of u
            degv = g->offsets[v + 1] - g->offsets[v]; // get the degree of v

            // we will not process (u,v) if u > v.
            if (u > v)
                continue;

            // create a pair storing the edge
            Pair edge;
            edge.first = u;  // first end is u
            edge.second = v; // second endpoint is v

            double edgeWt = 1 / double(degu * degv);                        // calculate the weight of the edge
                                                                            //            printf("The edge data: u = %" PRId64 ", v = %" PRId64 ", i = %" PRId64 ", edgeWt = %f, triangle weight = %f\n",u,v,i,edgeWt,triInfo->perEdge[i]);
            if (edgeStatus[i] != 'S' && triInfo->perEdge[i] < eps * edgeWt) // if the edge is not already on stack and triangle weight is less than eps*edgeWt, this edge should be cleaned
            {
                //                printf("The initial pushes: u = %" PRId64 ", v = %" PRId64 ", i = %" PRId64 ", edgeWt = %f, triangle weight = %f\n",u,v,i,edgeWt,triInfo->perEdge[i]);
                toDelete.push(edge);                // we push the edge onto the stack
                edgeStatus[i] = 'S';                // the edge is marked S, to denote it is in the stack
                edgeStatus[g->partnerMap[i]] = 'S'; // also mark the partner as 'S'
            }
        }

    // we have populated the set toDelete with the edges below the threshold
    // we will now call the real workhorse to do the actual edge deletions and cleaning

    deleteAndClean(g, edgeStatus, triInfo, &toDelete, eps); // call the function, and set ret to be the array indicating which edges remain

    // print out total triangle count
    printf("Total triangle count, post cleaning = %" PRId64 "\n", triInfo->total);

    // =========================================================================
    // START OF INSERTED COMPARISON BLOCK
    // =========================================================================
    // int64_t remaining_directed_edges = 0;
    // std::vector<std::pair<VertexIdx, VertexIdx>> surviving_edges;

    // // Loop through the graph and count/collect all edges marked 'Y' (survived)
    // for (VertexIdx u = 0; u < g->nVertices; u++) {
    //     for (EdgeIdx j = g->offsets[u]; j < g->offsets[u+1]; j++) {
    //         if (edgeStatus[j] == 'Y') {
    //             remaining_directed_edges++;
    //             VertexIdx v = g->nbors[j];
    //             // Only save u < v so we get an undirected edge list
    //             if (u < v) {
    //                 surviving_edges.push_back({u, v});
    //             }
    //         }
    //     }
    // }

    // printf("\n[GPU COMPARISON STATS]\n");
    // printf("Edges remaining after initialClean (Directed): %" PRId64 "\n", remaining_directed_edges);
    // printf("Edges remaining after initialClean (Undirected): %" PRId64 "\n\n", remaining_directed_edges / 2);

    // // Dump the surviving edges to a file so you can diff it against your GPU output
    // FILE* f_dump = fopen("amazon_cleaned_edges.txt", "w");
    // if (f_dump) {
    //     // Sort the edges so the file is deterministic and diff-able
    //     std::sort(surviving_edges.begin(), surviving_edges.end());
    //     for (const auto& edge : surviving_edges) {
    //         fprintf(f_dump, "%" PRId64 " %" PRId64 "\n", (int64_t)edge.first, (int64_t)edge.second);
    //     }
    //     fclose(f_dump);
    //     printf("Successfully dumped surviving edges to 'amazon_cleaned_edges.txt'\n");
    // } else {
    //     printf("Error: Could not open 'amazon_cleaned_edges.txt' for writing.\n");
    // }
    // =========================================================================
    // END OF INSERTED COMPARISON BLOCK
    // ===========

    // edgeStatus is updated by the previous function, so return it as the output
    return edgeStatus;
}

/* populateStats procedure: given a cluster object, this function just populates the properties of the cluster

Input:
    g: Pointer to a CGraph
    clus: Pointer to a cluster object

Output:
    void, but it changes the clus object

It simply goes over all edges incident to clus, and checks which of them are contained inside the cluster
*/
void populateStats(CGraph* g, Cluster* clus)
{
    clus->nVertices = (clus->vertices).size(); // call the size function to get the number of vertices
    clus->nEdges = 0; // initialize the value to zero
    clus->cut = 0; // initialize the value to zero
    for (auto iter = begin(clus->vertices); iter != end(clus->vertices); iter++) // iterate over the set of vertices given by clus->vertices
    {
        VertexIdx u = *iter; // dereference to get the vertex
        for (EdgeIdx i = g->offsets[u]; i < g->offsets[u+1]; i++) // loop over all the neighbors of the vertex u
        {
            VertexIdx v = g->nbors[i]; // get the vertex v
            if ((clus->vertices).find(v) != (clus->vertices).end()) // cluster has the neighbor v
                clus->nEdges++; // increment the number of edges
            else // this is a cut edge, since the other endpoint lies outside the cluster
                clus->cut++;
        }
    }
    clus->nEdges = clus->nEdges/2; // divide by two, because each edge is counted twice
}





/* disjointExtract procedure: this follows the RTRex procedure.
First, it computes the weighted triangle info, and then runs the initial clean operation. At this stage,
every edge has a triangle weight at least eps times the edge weight.
Then, the extract happens. It process vertices in increasing order of degree, and extract
clusters centered at the vertices. This is done repeatedly until the graph is empty.

Input:
    g: Pointer to a CGgraph
    eps: the decomposition parameter, which is the cleaning threshold
    cluster_map: As reference, keeps a mapping from vertex id to cluster

Output:
    A set of Cluster objects, that partition all vertices in the graph.
    There may be singleton clusters for vertices that do not produce anything non-trivial.

The algorithm works as follows. When a vertex u is processed, we look at all non-deleted
neighbors v. If the degree of v is at most (the degree of u)/eps, then we add it to the cluster. 
Next, we iterate over all *current* triangles involving two neighbors v, w of u. (If any triangle
has a removed edge, we ignore it.) Each such triangle involves a
vertex x that is two hops away from u. For all x, we compute the total triangle weight of such triangles.
This is done by iterating of all triangles (v,w,x) and updating an unordered map keyed by x.

Finally, we have a map that gives, for all (relevant x), the triangle weight incident on x by
two vertices in the neighborhood of u. (Note that there are precisely the triangles that are not
in the current cluster.) We sort the x's in decreasing order of this triangle weight.

We add these vertices x to the cluster one by one, keeping track of (i) the triangle weight incident on x
and (ii) the total triangle weight of the current cluster. We maximize the ratio: 
    
    (total triangle weight inside neighborhood of u and triangle weight contributed by two-hop vertices)/(total triangle weight of all vertices in the cluster)

We simply process ther vertices x in decreasing order (as said before), and find the maximum. So it's just a linear
loop over all the x vertices. 

At this point, we have created a new cluster. We put all edges inside the cluster into a stack, and send this
to deleteAndClean. This function will remove all these edges and clean to convergence. We repeat this loop
to get the next cluster, so on and so forth. At the end, all edges will have been deleted.

C. Seshadhri, Aug 2023
*/
vector<Cluster> disjointExtract(CGraph *g, double eps, std::map<VertexIdx, VertexIdx> &cluster_map)
{
    auto start = std::chrono::high_resolution_clock::now();

    vector<Cluster> decomposition;

    for (VertexIdx i = 0; i < g->nVertices; i++)
        cluster_map[i] = -1;

    WeightedTriangleInfo triInfo = commonNbr(g);

    auto stop = std::chrono::high_resolution_clock::now();
    cout << "Time for getting the weights: " << chrono::duration<double>(stop - start).count() << "s \n";
    start = std::chrono::high_resolution_clock::now();

    char *edgeStatus = initialClean(g, &triInfo, eps);

    stop = std::chrono::high_resolution_clock::now();
    cout << "Time for the initial clean: " << chrono::duration<double>(stop - start).count() << "s \n";
    start = std::chrono::high_resolution_clock::now();

    // Sort vertices by increasing degree
    Pair *deg_info = new Pair[g->nVertices];
    for (VertexIdx i = 0; i < g->nVertices; i++)
    {
        deg_info[i].first = i;
        deg_info[i].second = g->offsets[i + 1] - g->offsets[i];
    }
    std::sort(deg_info, deg_info + g->nVertices, pairCompareSecond);

    // CHANGE 1: int* instead of bool* — required for __sync_bool_compare_and_swap
    // and valid use with #pragma omp atomic. Initialized to 0 (not a two-hop vertex).
    double *triwgt = new double[g->nVertices]();
    bool *nbdBitMap = new bool[g->nVertices]();
    bool *clustered = new bool[g->nVertices]();
    int *twoHopBitMap = new int[g->nVertices](); // CHANGE 1

    int n_clusters = 0;

    for (VertexIdx i = 0; i < g->nVertices; i++)
    {
        VertexIdx u = deg_info[i].first;
        Count degu = deg_info[i].second;

        if (clustered[u])
            continue;

        Cluster next_clus;
        next_clus.vertices.insert(u);
        cluster_map[u] = n_clusters;
        clustered[u] = true;

        // ---------------------------------------------------------------
        // First neighbor pass: populate cluster and nbdBitMap. Serial, unchanged.
        // ---------------------------------------------------------------
        for (EdgeIdx j = g->offsets[u]; j < g->offsets[u + 1]; ++j)
        {
            if (edgeStatus[j] == 'N')
                continue;
            VertexIdx v = g->nbors[j];
            next_clus.vertices.insert(v);
            nbdBitMap[v] = true;
            cluster_map[v] = n_clusters;
            clustered[v] = true;
        }

        // ---------------------------------------------------------------
        // CHANGE 2: parallel two-hop enumeration.
        // reduction(+:internalWgt) gives each thread a private copy; the
        // master gets the sum after the implicit barrier at the end of the
        // parallel region.
        // CHANGE 3: per-thread localList avoids the O(n) full-array scan.
        // ---------------------------------------------------------------
        double internalWgt = 0;
        vector<VertexIdx> twoHopList; // filled from per-thread lists after parallel region

#pragma omp parallel reduction(+ : internalWgt) if (degu > 64)
        {
            vector<VertexIdx> localList; // private to each thread

#pragma omp for schedule(dynamic)
            for (EdgeIdx j = g->offsets[u]; j < g->offsets[u + 1]; ++j)
            {
                if (edgeStatus[j] == 'N')
                    continue;

                VertexIdx v = g->nbors[j];
                Count degv = g->offsets[v + 1] - g->offsets[v];

                if ((double)degv > (double)degu / eps)
                    continue;

                for (EdgeIdx k = j; k < g->offsets[u + 1]; ++k)
                {
                    if (edgeStatus[k] == 'N')
                        continue;

                    VertexIdx w = g->nbors[k];
                    Count degw = g->offsets[w + 1] - g->offsets[w];

                    if ((double)degw > (double)degu / eps)
                        continue;

                    EdgeIdx loc = g->getEdgeBinary(u, w);
                    if (loc == -1 || edgeStatus[loc] == 'N')
                        continue;

                    // triangle (u, v, w) confirmed; reduction var, no atomic needed
                    internalWgt += 1.0 / ((double)degu * (double)degv * (double)degw);

                    VertexIdx lower = (degv < degw) ? v : w;
                    VertexIdx higher = (degv < degw) ? w : v;

                    for (EdgeIdx ell = g->offsets[lower]; ell < g->offsets[lower + 1]; ell++)
                    {
                        if (edgeStatus[ell] == 'N')
                            continue;

                        VertexIdx x = g->nbors[ell];
                        Count degx = g->offsets[x + 1] - g->offsets[x];

                        if (x == lower || x == higher || x == u)
                            continue;

                        // CHANGE 6: original checks `loc == -1` here (always false
                        // at this point), missing the actual null-check for loc_hix.
                        EdgeIdx loc_hix = g->getEdgeBinary(higher, x);
                        if (loc_hix == -1 || edgeStatus[loc_hix] == 'N')
                            continue;

                        // triangle (v, w, x) confirmed
                        double wgt = 1.0 / ((double)degx * (double)degv * (double)degw);

                        if (nbdBitMap[x]) // x is already a direct neighbor of u
                        {
                            if (x > v && x > w)
                                internalWgt += wgt; // reduction var, no atomic needed
                            continue;
                        }

                        // x is a two-hop vertex.
                        // CHANGE 3: CAS flips twoHopBitMap[x] from 0 to 1 exactly
                        // once across all threads. The winning thread pushes x into
                        // its localList — so twoHopList has no duplicates after merge.
                        if (__sync_bool_compare_and_swap(&twoHopBitMap[x], 0, 1))
                            localList.push_back(x);

// CHANGE 4: triwgt[x] accumulates from multiple threads
#pragma omp atomic
                        triwgt[x] += wgt;
                    }
                }
            }
// omp for done. Merge this thread's localList into twoHopList.
#pragma omp critical
            twoHopList.insert(twoHopList.end(), localList.begin(), localList.end());

        } // parallel region ends; internalWgt reduced, twoHopList complete

        // ---------------------------------------------------------------
        // Candidate collection: O(|two_hop|), not O(n).
        // Resets twoHopBitMap and triwgt for the next cluster.
        // CHANGE 5: vector instead of raw new[]/delete[] — fixes use-after-free.
        // ---------------------------------------------------------------
        vector<wgtPair> candidates;
        candidates.reserve(twoHopList.size());
        double potentialWgt = 0;

        for (VertexIdx x : twoHopList)
        {
            wgtPair cp;
            cp.vertex = x;
            cp.wgt = triwgt[x];
            candidates.push_back(cp);
            potentialWgt += triwgt[x];
            triwgt[x] = 0;       // VERY IMPORTANT: reset for next cluster
            twoHopBitMap[x] = 0; // reset for next cluster
        }

        sort(candidates.begin(), candidates.end(), wgtCompareDecreasing);

        // ---------------------------------------------------------------
        // Ratio optimization: inherently sequential prefix scan, unchanged.
        // ---------------------------------------------------------------
        double ratio = internalWgt / (internalWgt + potentialWgt);
        int maxind = -1;
        double num_sum = 0, den_sum = 0;

        VertexIdx cand_size = (VertexIdx)candidates.size();
        for (VertexIdx index = 0; index < cand_size; index++)
        {
            num_sum += candidates[index].wgt;
            den_sum += triInfo.perVertex[candidates[index].vertex];

            double new_ratio = (internalWgt + num_sum) / (internalWgt + potentialWgt + den_sum);
            if (new_ratio > ratio)
            {
                ratio = new_ratio;
                maxind = (int)index;
            }
        }

        // Admit the best prefix. CHANGE 5: candidates is still valid here
        // (original delete[]'d it before this loop).
        for (VertexIdx index = 0; index <= (VertexIdx)maxind; index++)
        {
            VertexIdx next_vert = candidates[index].vertex;
            next_clus.vertices.insert(next_vert);
            cluster_map[next_vert] = n_clusters;
            clustered[next_vert] = true;
        }

        // Clear nbdBitMap for the next cluster (unchanged)
        for (EdgeIdx j = g->offsets[u]; j < g->offsets[u + 1]; ++j)
            nbdBitMap[g->nbors[j]] = false;

        populateStats(g, &next_clus);
        decomposition.push_back(next_clus);
        n_clusters++;

        // Remove all edges inside cluster and clean to convergence (unchanged)
        stack<Pair> toDelete;
        for (auto iterc = begin(next_clus.vertices); iterc != end(next_clus.vertices); iterc++)
        {
            VertexIdx z = *iterc;
            for (EdgeIdx j = g->offsets[z]; j < g->offsets[z + 1]; j++)
            {
                if (edgeStatus[j] != 'Y')
                    continue;
                Pair next;
                next.first = z;
                next.second = g->nbors[j];
                toDelete.push(next);
                edgeStatus[j] = 'S';
                edgeStatus[g->partnerMap[j]] = 'S';
            }
        }
        deleteAndClean(g, edgeStatus, &triInfo, &toDelete, eps);
    }

    stop = std::chrono::high_resolution_clock::now();
    cout << "Time for getting the clusters: " << chrono::duration<double>(stop - start).count() << "s \n";

    delete[] deg_info;
    delete[] triwgt;
    delete[] nbdBitMap;
    delete[] clustered;
    delete[] twoHopBitMap;
    // Note: edgeStatus is not freed here, consistent with original

    return decomposition;
}

/* expandClusters procedure: This procedure expands the existing clusters adding isolated vertices that are strongly connected to a cluster.

Input:
    g: Pointer to a CGgraph
    decomposition: The decomposition result from the triadic algorithm (or any decomposotion) of type vector<Cluster>
    cluster_map: A map from vertex to cluster id showing the cluster membership

Output:
    A set of Cluster objects, that partition all vertices in the graph.

For each non-clustered vertex the algorithm will find the cluster id of all the neighbours and add that vertex to the cluster with more neighbours
if there are at least threshold neighbours.

Daniel Paul Pena, Sep 2023
*/
vector<Cluster> expandClusters(CGraph* g, vector<Cluster> decomposition, std::map<VertexIdx, VertexIdx> cluster_map, int threshold, std::map<VertexIdx, double>& scores)
{
    VertexIdx size = decomposition.size();
    for (VertexIdx i = 0; i < size; i++) { //loop over all vertices

        if(decomposition[i].nVertices > 1){
            continue;
        }
        VertexIdx v  = *decomposition[i].vertices.cbegin(); //v is the only vertex of a cluster of size 1
        std::map<VertexIdx, VertexIdx> counts;

        VertexIdx best = -1;
        VertexIdx maxScore = threshold; //We require at least threshold neighbours in the cluster
        //the vertex is not clustered
        EdgeIdx number_edges = g->offsets[v+1] - g->offsets[v];
        for (EdgeIdx j=g->offsets[v]; j < g->offsets[v+1]; ++j) // looping over neighbors of v
        {
            VertexIdx u = g->nbors[j];
            if (cluster_map[u] == -1){
                continue;
            }
            VertexIdx cluster_id = cluster_map[u]; //neighbour u belongs to cluster cluster_id
            if (counts.count(cluster_id) == 0) {
                counts[cluster_id] = 0;
            }
            counts[cluster_id]++;

            //Update the best neighboring cluster if neccesary
            if (counts[cluster_id] > maxScore){
                best = cluster_id;
                maxScore = counts[cluster_id];
            }
        }
        //best is the closest neighbour for vertex v, check it is not -1
        if (best == -1){
            //In this case no cluster went above the threshold, we keep the vertex in its own cluster
            continue;
        }
        //we need to add v to the cluster best and remove it from the original cluster
        decomposition[best].vertices.insert(v);
        decomposition[i].vertices.erase(v);
        scores[v] = (double) maxScore/number_edges;
    }
    //to finish we recompute the stats of all the clusters
    for (VertexIdx i = 0; i < size; i++) {
        populateStats(g, &decomposition[i]);
    }

    return decomposition;

}

/* connectedComp procedure: This finds the connected components of the graph, and stores them as a list of clusters.

Input:
    g: Pointer to a CGgraph

Output:
    A set of Cluster objects, that partition all vertices in the graph.

The procedure is a standard bfs.

C. Seshadhri, Aug 2023
*/

vector<Cluster> connectedComp(CGraph* g)
{
    vector<Cluster> decomposition;

    bool* visited = new bool[g->nVertices]; // a bitmap storing the visited status
    for (VertexIdx u=0; u < g->nVertices; u++) // initializing the visited array
        visited[u] = false;

    queue<VertexIdx> q; // the initially empty queue of vertices

    for (VertexIdx u=0; u < g->nVertices; u++) // loop over the vertices
    {
        if (visited[u]) // u is already visited
            continue; // move on to the next vertex

        Cluster next_clus; // start the next cluster
        next_clus.vertices.insert(u); // insert u into the cluster
        visited[u] = true; // mark u as visited
        q.push(u); // push u into the queue to initialize

        while(!q.empty()) // while the queue is non-empty
        {
            VertexIdx next = q.front(); // getting the next vertex of the q
            q.pop(); // also remove the element
            for (EdgeIdx j = g->offsets[next]; j < g->offsets[next+1]; j++) // loop over the neighbors of j
            {
                VertexIdx nbr = g->nbors[j]; // get the neighbor under question
                if (!visited[nbr]) // the neighbor has not been visited
                {
                    q.push(nbr); // add the neighbor to the q
                    visited[nbr] = true; // we have visited the neighbor
                    next_clus.vertices.insert(nbr); // get the neighbor into the cluster
                }
            }
        }
        // we have constructed the cluster, in the set next_clus.
        // We now write out the basic stats of the cluster, done in a separate function. Note that the function modifies the stats in the next_clus object.
        populateStats(g,&next_clus); 

        // We now add it the cluster to the decomposition.
        decomposition.push_back(next_clus); // replaced by push_back as it is a vector
    }

    return decomposition;

}















#endif


