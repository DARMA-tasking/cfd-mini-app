import math
import vtk

class Mesh:
    def __init__(self,dx,dy,h):
        self.O=[0,0]
        self.dim=[int(dx),int(dy)]
        self.size=float(h)
        self.n_points=(dx+1) * (dy+1)
        self.n_cells=dx * dy
        self.h = h

        self.n_cells_x = dx
        self.n_cells_y = dy

        self.n_points_x = dx+1
        self.n_points_y = dy+1

        self.cell_scalars = [0. for _ in range(self.n_cells)]
        self.cell_scalars_name='cell scalar'
        self.cell_vectors=[[0.,0.] for _ in range(self.n_cells)]
        self.cell_vectors_name='cell vector'

        # Point vectors have an x and y component
        self.point_vectors=[[0.,0.] for _ in range(self.n_points)]
        self.point_vectors_name='point vector'
        self.point_scalars=[0. for _ in range(self.n_points)]
        self.point_scalars_name='point scalar'

    ####################
    # Basic
    ####################

    def set_origin(self,x,y):
        self.O=[x,y]

    def index_to_cartesian(self,k,n,nmax):
        if k<0 or k>=nmax:
            return((None,None))
        else:
            (q,r)=divmod(k,n)
            return((r,q))

    def cartesian_to_index(self,i,j,ni,nj):
        if i<0 or i>=ni or j<0 or j>=nj:
            return(None)
        else:
            return(j * ni + i)

    def print(self):
        self.print_meta()
        print("point vectors: {}".format(self.point_vectors))
        print("cell scalars: {}".format(self.cell_scalars))

    def print_meta(self):
        print("origin: ({}; {}) size: {} dimensions: ({}; {})".format(
        self.O[0],self.O[1],self.size,self.dim[0],self.dim[1]))

    def i_max(self): # TODO: rename to i_point_max
        return self.dim[0]

    def j_max(self):
        return self.dim[1]


    ####################
    # Cells
    ####################

    def get_cell_scalars(self):
        return(self.cell_scalars)

    def set_cell_scalars_name(self,name):
        self.cell_scalars_name=name

    def get_cell_scalar(self,i,j):
        k=self.cartesian_to_index(i,j,self.dim[0],self.dim[1])
        if k==None:
            return(math.nan)
        else:
            return(self.cell_scalars[k])

    def set_cell_scalar(self,i,j,scalar):
        k=self.cartesian_to_index(i,j,self.dim[0],self.dim[1])
        if k!=None:
            self.cell_scalars[k]=scalar

    def initialize_cell_scalars_by_coordinate_function(self, fct):
        y = self.O[1] + 0.5 * self.size
        for j in range(self.dim[1]):
            x = self.O[0] + 0.5 * self.size
            for i in range(self.dim[0]):
                print(x,y)
                self.set_cell_scalar(i,j,fct(x,y))
                x += self.size
            y += self.size


    ####################
    # Points
    ####################

    def set_point_vectors_name(self,name):
        self.point_vectors_name=name

    def get_point_vectors(self):
        return(self.point_vectors)

    def get_point_vector(self,i,j):
        k=self.cartesian_to_index(i,j,self.dim[0]+1,self.dim[1]+1)
        if k==None:
            return(math.nan)
        else:
            return(self.point_vectors[k])

    def set_point_vector(self,i,j,c):
        k=self.cartesian_to_index(i,j,self.dim[0]+1,self.dim[1]+1)
        if k!=None:
            self.point_vectors[k]=c

    def set_point_vector_u(self, i, j, u):
        k = self.cartesian_to_index(i, j, self.dim[0]+1, self.dim[1]+1)
        if k != None:
            (self.point_vectors[k])[0] = u

    def set_point_vector_v(self, i, j, v):
        k = self.cartesian_to_index(i, j, self.dim[0]+1, self.dim[1]+1)
        if k != None:
            self.point_vectors[k][1] = v

    def get_point_vector_u(self, i, j):
        return self.get_point_vector(i,j)[0]

    def get_point_vector_v(self, i, j):
        return self.get_point_vector(i,j)[1]

    def initialize_point_vectors_by_coordinate_component_function(self, f):
        y = self.O[1]
        for j in range(self.dim[1]+1):
            x = self.O[0]
            for i in range(self.dim[0]+1):
                self.set_point_vector(i,j,f(x,y))
                x += self.size
            y += self.size


    ####################
    # VTK
    ####################

    def write_vtk(self, file_name):
        ug = vtk.vtkUniformGrid()
        nx=self.dim[0]
        ny=self.dim[1]
        ug.SetDimensions(nx+1, ny+1, 1)
        ug.SetOrigin(self.O[0] ,self.O[0], 0)
        ug.SetSpacing(self.size, self.size, 0)

        # Create cell centered scalar field
        cell_data = vtk.vtkDoubleArray()
        cell_data.SetNumberOfComponents(1)
        cell_data.SetName(self.cell_scalars_name)
        cell_data.SetNumberOfValues(nx * ny)
        for j in range(ny):
            for i in range(nx):
                cell_data.SetValue(j*nx + i, self.get_cell_scalar(i,j))
        ug.GetCellData().SetScalars(cell_data)

        # Create point centered vector field
        point_data = vtk.vtkDoubleArray()
        point_data.SetNumberOfComponents(3)
        point_data.SetName(self.point_vectors_name)
        nxp1 = nx + 1
        nyp1 = ny + 1
        point_data.SetNumberOfTuples(nxp1 * nyp1)
        for j in range(nyp1):
            for i in range(nxp1):
                point_data.SetTuple3(j*nxp1 + i, *(self.get_point_vector(i,j)+[0]))
        ug.GetPointData().SetVectors(point_data)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(file_name)
        writer.SetInputData(ug)
        writer.Write()
