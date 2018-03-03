function [mfWD, mnNumConns, fEUnitary, fIUnitary] = DiscretiseWeightMatrix(mfW, nMaxConnections)

fEUnitary = max(mfW(:)) ./ nMaxConnections;
fIUnitary = max(-mfW(:)) ./ nMaxConnections;
vfExcValues = linspace(0, fEUnitary, nMaxConnections);
vfInhValues = linspace(0, fIUnitary, nMaxConnections);

mnEConns = round(mfW ./ fEUnitary) .* (mfW > 0);
mnIConns = -round(mfW ./ fIUnitary) .* (mfW < 0);

mnNumConns = mnEConns - mnIConns;

mfWD = mnEConns .* fEUnitary - mnIConns .* fIUnitary;

