function out= sigmoid(Z)
	out = 1 ./ (1 + exp(-Z));
end