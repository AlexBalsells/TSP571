import gurobipy as grb
from gurobipy import GRB

#objective coefficients
c=[5, 2]

#create model
Model = grb.Model()

#set heuristic and cuts attributes to zero
#Model.setParam("Cuts", 0)
#Model.setParam("Heuristics", 0)

#define x variables and set objective values
x = Model.addVars(2, obj=c, lb= 0.0, vtype=GRB.CONTINUOUS, name="x")


#set objective function sense
Model.modelSense = GRB.MAXIMIZE

#add constraints to model
Model.addConstr(5*x[0] + 2*x[1] <= 60.0)
Model.addConstr(2*x[0] + x[1] <= 25.0)

#Let Gurobi know that the model has changed
Model.update()

#write out the lp in a lp-file
Model.write("shrekpy.lp")

#optimize model
Model.optimize()

#if status comes back as optimal (value=2) then print out ony nonzero solution values
if Model.status == 2:
   line="Total profit: "+str(Model.objVal)+"\n"
   print(line)

   for j in range(2):
       if x[j].x > 1e-9:
           #print('product %s produce quantity \n' % j)
           line = "You should produce "+str(x[j].x)+" of product "+str(j)+"\n"
           print(line)

