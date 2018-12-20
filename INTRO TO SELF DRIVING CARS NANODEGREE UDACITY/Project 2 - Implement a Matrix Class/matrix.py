import math
from math import sqrt
import numbers

def zeroes(height, width):
        """
        Creates a matrix of zeroes.
        """
        g = [[0.0 for _ in range(width)] for __ in range(height)]
        return Matrix(g)

def identity(n):
        """
        Creates a n x n identity matrix.
        """
        I = zeroes(n, n)
        for i in range(n):
            I.g[i][i] = 1.0
        return I

class Matrix(object):

    # Constructor
    def __init__(self, grid):
        self.g = grid
        self.h = len(grid)
        self.w = len(grid[0])

    #
    # Primary matrix math methods
    #############################
 
    def determinant(self):
        """
        Calculates the determinant of a 1x1 or 2x2 matrix.
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate determinant of non-square matrix.")
        if self.h > 2:
            raise(NotImplementedError, "Calculating determinant not implemented for matrices largerer than 2x2.")
        
        # TODO - your code here
        determinant = 0
        if self.h == 1:
            determinant = self.g[0][0]
        else:
            determinant = self.g[0][0] * self.g[1][1] - self.g[1][0] * self.g[0][1]
        return determinant

    def trace(self):
        """
        Calculates the trace of a matrix (sum of diagonal entries).
        """
        if not self.is_square():
            raise(ValueError, "Cannot calculate the trace of a non-square matrix.")

        # TODO - your code here
        
        return sum(self.g[i][i] for i in range(self.h))

    def inverse(self):
        """
        Calculates the inverse of a 1x1 or 2x2 Matrix.
        """
        if not self.is_square():
            raise(ValueError, "Non-square Matrix does not have an inverse.")
        if self.h > 2:
            raise(NotImplementedError, "inversion not implemented for matrices larger than 2x2.")

        # TODO - your code here
        determinant = self.determinant()
        new_matrix = []   
        if self.h == 1:
            return Matrix([[1/self.g[0][0]]])
        else:
            multiplier = 1 / determinant
            row_1 = [multiplier*self.g[1][1], -multiplier*self.g[0][1]]
            row_2 = [-multiplier*self.g[1][0], multiplier*self.g[0][0]]
            new_matrix.append(row_1)
            new_matrix.append(row_2)

        return Matrix(new_matrix)

    def T(self):
        """
        Returns a transposed copy of this Matrix.
        """
        # TODO - your code here
         
        return Matrix([[self.g[j][i] for j in range(self.h)] for i in range(self.w)])

    def is_square(self):
        return self.h == self.w

    #
    # Begin Operator Overloading
    ############################
    def __getitem__(self,idx):
        """
        Defines the behavior of using square brackets [] on instances
        of this class.

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > my_matrix[0]
          [1, 2]

        > my_matrix[0][0]
          1
        """
        return self.g[idx]

    def __repr__(self):
        """
        Defines the behavior of calling print on an instance of this class.
        """
        s = ""
        for row in self.g:
            s += " ".join(["{} ".format(x) for x in row])
            s += "\n"
        return s

    def __add__(self,other):
        """
        Defines the behavior of the + operator
        """
        if self.h != other.h or self.w != other.w:
            raise(ValueError, "Matrices can only be added if the dimensions are the same") 
        #   
        # TODO - your code here
        #
        new_grid = []
        for i in range(self.h):
            new_row = []
            for j in range(self.w):
                value1 = self.g[i][j]
                value2 = other.g[i][j]
                value3 = value1 + value2
                new_row.append(value3)
            new_grid.append(new_row)
        return Matrix(new_grid)

    def __neg__(self):
        """
        Defines the behavior of - operator (NOT subtraction)

        Example:

        > my_matrix = Matrix([ [1, 2], [3, 4] ])
        > negative  = -my_matrix
        > print(negative)
          -1.0  -2.0
          -3.0  -4.0
        """
        #   
        # TODO - your code here
        #
        new_2d = []
        for row in self.g:
            new_row = []
            for value in row:
                new_value = -1 * value
                new_row.append(new_value)
            new_2d.append(new_row)
        return Matrix(new_2d)

    def __sub__(self, other):
        """
        Defines the behavior of - operator (as subtraction)
        """
        #   
        # TODO - your code here
        #
        

        return self + -other

        

    def __mul__(self, other):
        """
        Defines the behavior of * operator (matrix multiplication)
        """
        #   
        # TODO - your code here
        #
        if self.w != other.h:
            raise (ValueError, "Matrices can only be multiplied if the width of  A  is equal to the height of B")
        other_Transpose = []
        new_matrix = []
        for i in range(other.w):
            temporary = []
            for j in range(other.h):
                temporary.append(other[j][i])
            other_Transpose.append(temporary)    
        for i in range(len(self.g)):
            temporary = []
            for j in range(len(other_Transpose)):
                value = 0
                for k in range(len(self.g[0])):
                    value = value + self.g[i][k] * other_Transpose[j][k]
                temporary.append(value) 
            new_matrix.append(temporary)    
        return Matrix(new_matrix)

    def __rmul__(self, other):
        """
        Called when the thing on the left of the * is not a matrix.

        Example:

        > identity = Matrix([ [1,0], [0,1] ])
        > doubled  = 2 * identity
        > print(doubled)
          2.0  0.0
          0.0  2.0
        """
        if isinstance(other, numbers.Number):
            pass
            new_matrix = []
            for row in range(len(self.g)):
                new_row = []
                for col in range(len(self.g[0])):
                    new_row.append(other * self[row][col])
                new_matrix.append(new_row)
            return Matrix(new_matrix)
            
            
         
            
           
            