# Imdescrip -- a collection of tools to extract descriptors from images.
# Copyright (C) 2013  Daniel M. Steinberg (d.steinberg@acfr.usyd.edu.au)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" Generic (abstract base) descriptor object to inherit. """

import abc

class Descriptor:
    """ This class is the abstract base class for image descriptor objects. 
    
        If this interface is followed, the image descriptor objects should work
        with the batch image descriptor extractors.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def extract (self, image):
        """ Method required for actual descriptor extraction. 
        
            This method should accept an image file name and should return an
            object/array which is the actual feature.
        """
        pass


