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

""" A simple progress bar to reduce this package's dependencies. """

import sys

class Progress ():
    """ A very simple progress bar e.g. 'Title : [========            ] 80/200'.

    This class implements a fairly rudimentary progress bar, mostly to reduce
    the dependencies on external libraries which do a superset of what is
    required.

    The way this class is typically used is

        #Set up code
        ...
        progbar = Progress(max_val, ...)

        # Do something
        for i, it in enumerate(iterable)
            ...
            progbar.update(i)

        # Done and clean up
        progbar.finished()

    Arguments:
        max_val: int, the maximum value of iterations/items to iterate.
        cwidth: int, the with of the progress bar in characters (the number of 
            '=' characters).
        title: str, a title to show before the progress bar, this is optional.
        verbose: bool, whether to show any progress (True) or to be silent
            (False).

    """

    def __init__ (self, max_val, cwidth=20, title=None, verbose=True):

        self.max_val = max_val
        self.cwidth  = cwidth
        self.verbose = verbose

        if title is None:
            self.title = ''
        else:
            self.title = title + ' : ' 

        self.update(0)


    def update (self, val):
        """ Update the progress bar.

        This will draw and update the progress bar to reflect the current
        iterations/iterables done. That is current_value/max_value.

        Arguments:
            val: int, the current iteration/iterable object being operated on.
                This must be in the range [0, max_val] otherwise an error is
                raised.

        """
        
        if self.verbose == False:
            return

        if (val < 0) or (val > self.max_val):
            raise ValueError('Argument val is out of range!')

        pcomp = float(val) / self.max_val
        nbars = int(round(pcomp * self.cwidth))
        nspace = self.cwidth - nbars

        sys.stdout.write('\r' + self.title + '[' + '=' * nbars + ' ' * nspace 
                         + '] ' + str(val) + '/' + str(self.max_val))
        sys.stdout.flush()


    def finished (self):
        """ Make sure the progress bar says all iterations have been completed.

            This essentially just wraps Progress.update(max_val) and tidies up
            the command line.
        """

        if self.verbose == False:
            return

        self.update(self.max_val)
        
        sys.stdout.write('\n')
        sys.stdout.flush()
