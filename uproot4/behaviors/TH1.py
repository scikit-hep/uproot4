# BSD 3-Clause License; see https://github.com/scikit-hep/uproot4/blob/master/LICENSE

from __future__ import absolute_import

import numpy

import uproot4.models.TArray


class TH1(object):
    @property
    def np(self):
        xaxis = self.member("fXaxis")

        xaxis_fNbins = xaxis.member("fNbins")
        xedges = numpy.empty(xaxis_fNbins + 3, dtype=numpy.float64)
        xedges[0] = -numpy.inf
        xedges[-1] = numpy.inf

        xaxis_fXbins = xaxis.member("fXbins", none_if_missing=True)
        if xaxis_fXbins is None or len(xaxis_fXbins) == 0:
            xedges[1:-1] = numpy.linspace(
                xaxis.member("fXmin"), xaxis.member("fXmax"), xaxis_fNbins + 1
            )
        else:
            xedges[1:-1] = xaxis_fXbins

        for base in self.bases:
            if isinstance(base, uproot4.models.TArray.Model_TArray):
                values = numpy.array(base, dtype=base.dtype.newbyteorder("="))
                break

        return values, xedges
