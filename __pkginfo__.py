# -*- coding: ISO-8859-1 -*-
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

modname = 'hmm'
distname = 'logilab-hmm'

numversion = [0, 5, 0]
version = '.'.join([str(num) for num in numversion])


license = 'GPL'
copyright = '''Copyright © 2001-2003 LOGILAB S.A. (Paris, FRANCE).
http://www.logilab.fr/ -- mailto:contact@logilab.fr'''

short_desc = "Hidden Markov Models in Python"
long_desc = """Hidden Markov Models in Python
Implementation based on _A Tutorial on Hidden Markov Models and Selected
Applications in Speech Recognition_, by Lawrence Rabiner, IEEE, 1989.
This module uses numeric python multyarrays to improve performance and
reduce memory usage."""

author = "Alexandre Fayolle"
author_email = "Alexandre.Fayolle@logilab.fr"

web = "http://www.logilab.org/projects/%s" % modname
ftp = "ftp://ftp.logilab.org/pub/%s" % modname
mailinglist = "http://lists.logilab.org/mailman/listinfo/ai-projects"


subpackage_of = 'logilab'

try:
    from numpy.distutils.extension import Extension
    ext_modules = [Extension('hmm_ops', ['_hmmf.f90'],
                             libraries=['gfortran'])]
except:
    pass
    
