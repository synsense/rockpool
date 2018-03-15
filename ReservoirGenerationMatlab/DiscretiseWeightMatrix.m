function [mfWD, mnNumConns, fEUnitary, fIUnitary] = DiscretiseWeightMatrix(mfW, nMaxConnections)

% DiscretiseWeightMatrix - FUNCTION Discretise a weight matrix by strength
%
% Usage: [mfWD, mnNumConns, fEUnitary, fIUnitary] = DiscretiseWeightMatrix(mfW, nMaxConnections)
%
% `mfW` is an arbitrary real-valued weight matrix. `nMaxConnections` is the
% integer maximum number of synaptic connections that may be made between
% two neurons. Excitatory and inhibitory weights will be discretised
% separately.
%
% `mfWD` will be a discretised version of `mfW`, such that all weights
% consist of up to `nMaxConnections` connections.
%
% `mnNumConns` will be a matrix the same size as `mfWD`, with each element
% indicating the integer number of unitary synaptic connections made
% between two neurons. Inhibitory connections are negative integers.
%
% `fEUnitary` and `fIUnitary` will be scalar values containing the absolute
% strength of unitary excitatory and inhibitory synapses, respectively.

fEUnitary = max(mfW(:)) ./ nMaxConnections;
fIUnitary = max(-mfW(:)) ./ nMaxConnections;
vfExcValues = linspace(0, fEUnitary, nMaxConnections);
vfInhValues = linspace(0, fIUnitary, nMaxConnections);

mnEConns = round(mfW ./ fEUnitary) .* (mfW > 0);
mnIConns = -round(mfW ./ fIUnitary) .* (mfW < 0);

mnNumConns = mnEConns - mnIConns;

mfWD = mnEConns .* fEUnitary - mnIConns .* fIUnitary;

