function mfWeights = SparseTernaryConnMatrix(nNumNeurons, nNumExcConn, nNumInhConn)

mfWeights = zeros(nNumNeurons);

for nSource = 1:nNumNeurons
    vnDest = randsample(nNumNeurons, nNumExcConn + nNumInhConn);
    mfWeights(vnDest(1:nNumExcConn), nSource) = 1;
    mfWeights(vnDest((1:nNumInhConn) + nNumExcConn), nSource) = -1;
end
