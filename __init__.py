# -*- coding: iso-8859-1 -*-
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
""" Copyright (c) 2002-2003 LOGILAB S.A. (Paris, FRANCE).
http://www.logilab.fr/ -- mailto:contact@logilab.fr  
"""

__revision__ = "$Id: __init__.py,v 1.4 2004-11-07 15:27:04 nico Exp $"

import hmm
HMM = hmm.HMM
HMM_PY = HMM
try:
    import hmmc
    HMM = HMM_C = hmmc.HMM_C
    
except ImportError:
    pass

try:
    import hmmf
    HMM = HMM_F = hmmf.HMM_F
except ImportError:
    pass
