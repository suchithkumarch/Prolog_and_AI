% Prolog and AI - PL project 
% by Imt2018011 - Arem Venkata Karthik Reddy,
%    Imt2018019 - Chebrolu Suchith Kumar &
%    Imt2018083 - Venkata Sai Gopal Sundari.

% This project has two basic linear models implemented in prolog.
% 1. Linear Classifier 2. Perceptron

:-use_module(library(clpfd)).

% A and B are numbers

my_multiply(A, B, C):-
	C is A*B.

my_plus(A, B, C):-
	C is A+B.

my_subtract(A, B, C):-
	C is A-B.
	
% populate a list of Length with a element .

populate_list(Ele,Length,ListE):-
	length(List,Length),findall(X,(member(X,List),X=Ele),ListE).	

% X and Y are vectors

vector_addition(X,Y,Z):-
	maplist(my_plus,X,Y,Z).

vector_subtraction(X,Y,Z):-
	maplist(my_subtract,X,Y,Z).

% S is a scalar and X is a vector.

scalar_multiplication(S,X,SX):-
	maplist(my_multiply(S),X,SX).

vector_length(V,L):-
	maplist(my_multiply,V,V,Squared),
	sumlist(Squared,SumSquare),
	L is SumSquare^0.5.

dot_product(V1,V2,Dot):-
	maplist(my_multiply,V1,V2,VMul),
	sumlist(VMul,Dot).

my_append(A, B, C):-
	append(B, A, C).

 
% W is a vector and T is a scalar.
% To classifiy new example, we can take dotprouct with w and check if over T,
% if so classify as positive.

get_prediction(Vector,classifier(W,T)):-
	dot_product(W,Vector,D),
	D>T.

test_new_point:-
	write('Enter X co-ordinate:'),nl,
	read(X),
	write('Enter Y co-ordinate:'),nl,
	read(Y),	
	data_classifier(Lc,Ptrn),
	write('Linear classifier Prediction:'),nl,
	(get_prediction([X,Y],Lc)->Bool1=true;Bool1=false),
	write(Bool1),nl,
	write('Perceptron Prediction:'),nl,
	(get_prediction([X,Y],Ptrn)->Bool2=true;Bool2=false),
	write(Bool2).

% Ps is a matrix, where each row is a pos example. Similar for Ns.

data_classifier(Lin_classifier,Perceptrn):-
		All=[[1.5998426,0.52985437,1],
                [0.25065517,1.30425162,1],
                [0.76148911,0.60419602,1],
                [0.75591032,-0.78994764,1],
                [1.63605539,0.9225655,1],
                [2.70520379,0.93285704,1],
                [1.82870703,2.34804646,1],
                [-0.08549264,0.99868399,1],
                [0.44906531,0.90555838,1],
                [0.49966187,1.59299327,1],
                [1.00003726,-0.13822094,1],
                [1.67943676,1.25283262,1],
                [-1.00158649,2.73839505,1],
                [3.32539035,-0.39289509,1],
                [2.17885898,0.05984356,1],
                [1.85977529,0.76782626,1],
                [1.34470454,0.18312675,1],
                [0.5974872,0.1228956,1],
                [-1.52394333,-1.24558361,-1],
                [-2.48452861,-1.91070328,-1],
                [-1.04605257,-2.55270759,-1],
                [1.02370408,-1.67944911,-1],
                [-0.80492117,-1.49215482,-1],
                [-1.64954319,-3.41635041,-1],
                [-2.35543276,-0.37750433,-1],
                [-0.32384031,-2.08235145,-1],
                [-1.56576954,-1.22018985,-1],
                [-1.27853841,-1.28469686,-1],
                [-1.97696119,0.23717806,-1],
                [-1.78965834,-1.09026084,-1]],
	linear_classifier(All,Lin_classifier),
	perceptron(All,Perceptrn).

linear_classifier(All,classifier(W,T)):-
	findall(Instance,(member(InstanceC,All),append(Instance,[1|[]],InstanceC)),Ps),
	findall(Instance,(member(InstanceC,All),append(Instance,[-1|[]],InstanceC)),Ns),
	write('Training the linear classifier....'),nl, 
	Ps =[OneExample|_],
	length(OneExample,NumberOfFeatures),
	populate_list(0,NumberOfFeatures,Zerovector),
	length(Ps,NP),
	length(Ns,NN),
	foldl(vector_addition,Ps,Zerovector,SumvectorPos),
	foldl(vector_addition,Ns,Zerovector,SumvectorNegs),
	OneOverNp is 1/NP,
	OneOverNN is 1/NN,
	scalar_multiplication(OneOverNp,SumvectorPos,CenterOfMassP),
	scalar_multiplication(OneOverNN,SumvectorNegs,CenterOfMassN),
	vector_subtraction(CenterOfMassP,CenterOfMassN,W), %Subtracting P-N =W
	vector_length(CenterOfMassP,LengthP),
	vector_length(CenterOfMassN,LengthN),
	my_multiply(LengthP,LengthP,PSquared),
	my_multiply(LengthN,LengthN,NSquared),
	T is (PSquared - NSquared)/2,
	write('The W, T values of the linear classifier are '), 
	write(W),
	write(" & "),
	write(T),nl.

% Gradient Descent Approach:

perceptron(All, classifier(W,0)):-
	write('Training the perceptron....'),nl,
	All =[HeadEle|_],
	length(HeadEle, NoColumns),
	my_subtract(NoColumns, 1, NumofFeatures),
	populate_list(0, NumofFeatures,Zerovector),
	convergedloop(All,Zerovector,W),
	write('The W value of the perceptron is '),
	write(W),nl.

convergedloop(All,Inputvector,OutputVector):-
	forloop(All,Inputvector,UpdateVector),
	check_converged(All,Inputvector,UpdateVector,OutputVector).

check_converged(All,W1,W2,W2):-W1=W2.
check_converged(All,W1,W2,W3):-
	dif(W1,W2),
	convergedloop(All,W2,W3).

forloop([],W,W).
forloop(All,W1,WFin):-
    All =[HeadEle|TailEle],
    update_weight(HeadEle,W1,W2),
    forloop(TailEle,W2,WFin).

misclassified(W,Datapoint,Bool):-
	append(Data,[Label|[]],Datapoint),
	dot_product(Data,W,Dot_prod),
	my_multiply(Dot_prod,Label,Result),
	(Result=<0->Bool=true;Bool=false).

learning_rate(0.1).

update_weight(Datapoint,W1,W2):-
	misclassified(W1,Datapoint,true),
	learning_rate(N),
	append(Data,[Label|[]],Datapoint),
	Value is N *Label,
	scalar_multiplication(Value,Data,Data_Value),
	vector_addition(W1,Data_Value,W2).

update_weight(Datapoint,W,W):-
	misclassified(W,Datapoint,false).
