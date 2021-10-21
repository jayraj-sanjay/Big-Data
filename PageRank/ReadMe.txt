Aim: To Calculate the importance of each airport using Page Rank Method
Data: List of originId and destinationId of airports from https://transtats.bts.gov/
Procedure:
-Calculated the Inlink and Outlink values of each airport.
-Set the initial page rank to be 10 for each airport.
-Iteratively updated the values for pageRank using the formula:

PR(x) = a * 1/N + [(1 - a) × Summation of i =(1 to n)=> (PR(ti)/C(ti))

where a = 0.15 and x is a page with inlinks from t1, t2, . . . , tn, C(t) is the out-degree of t, and
N is the total number of nodes in the graph.

Result: Airports arranged in the descending of their Page Rank Values.
