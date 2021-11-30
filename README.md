# TravelingSP

Dear Dr. Hicks,

We were able to succesfully implement branch and bound for solving IPPs. The file to solve is called, "branchbound", and it is a Jupyter Notebook. Follow instructions in notebook.

INPUT: the name of the file ie "att48.txt"
Possible errors: if the .txt file has several indents, initialize with post an error since get_df returned a dataframe of NaNs. To rememedy this, edit the .txt file and remove the superfluous indents. Notice, our .txt files in Cities/ have received this preprocessing step.

main_TSP: This function solves the TSP (without subtour elimination constraints) and prints the best_cost

create_G_sol: This function prints 

In the repo, the folder Cities/ contains the .txt files we want to solve the TSP on as well as houses the saved figures of our solutions. Ignore all other files/folders.