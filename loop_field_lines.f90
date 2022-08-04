PROGRAM LOOP_FIELD_LINES

USE HDF5

IMPLICIT NONE

DOUBLE PRECISION, PARAMETER  :: pi=acos(-1.0)
!***********************************************************************
INTEGER, PARAMETER :: tmin=0
INTEGER, PARAMETER :: tmax=2
INTEGER, PARAMETER :: FDUMP=10
INTEGER, PARAMETER           :: NX=80,NY=80,NZ=80
DOUBLE PRECISION, PARAMETER  :: rmin=1.0
DOUBLE PRECISION, PARAMETER  :: rmax=8.0
DOUBLE PRECISION             :: rpml=0.9*rmax
INTEGER, PARAMETER           :: NT=10000
DOUBLE PRECISION             :: dt=0.006059516324485058
INTEGER, PARAMETER           :: NGEOR = 1, NGEOTH = 6, NGEOPH = 8
!***********************************************************************
DOUBLE PRECISION, DIMENSION(1:NX,1:NY,1:NZ)   :: Bx,By,Bz
DOUBLE PRECISION, DIMENSION(1:NX)             :: x
DOUBLE PRECISION, DIMENSION(1:NY)             :: y
DOUBLE PRECISION, DIMENSION(1:NZ)             :: z
DOUBLE PRECISION, DIMENSION(1:NGEOR)          :: r0t
DOUBLE PRECISION, DIMENSION(1:NGEOTH)         :: th0t
DOUBLE PRECISION, DIMENSION(1:NGEOPH)         :: ph0t
INTEGER                                       :: it,it0
INTEGER                                       :: numline
INTEGER                                       :: i,j,NTF
INTEGER                                       :: ir,ith,iph
INTEGER                                       :: irt,itht,ipht
!***********************************************************************
DOUBLE PRECISION                            :: r0,theta0,phi0,ph0tr
DOUBLE PRECISION                            :: x2,y2,z2,x1,y1,z1
DOUBLE PRECISION                            :: k1x,k2x,k3x,k4x
DOUBLE PRECISION                            :: k1y,k2y,k3y,k4y
DOUBLE PRECISION                            :: k1z,k2z,k3z,k4z
DOUBLE PRECISION                            :: rt,tht,pht
DOUBLE PRECISION                            :: Brt,Btht,Bpht
DOUBLE PRECISION                            :: Br0,Bth0,Bph0
DOUBLE PRECISION                            :: Bxt,Byt,Bzt,Bt
DOUBLE PRECISION                            :: sico,sisi,co
DOUBLE PRECISION                            :: ds
DOUBLE PRECISION                            :: trilinear_bfield
!***********************************************************************
DOUBLE PRECISION, DIMENSION(1:NT)           :: xt,yt,zt
DOUBLE PRECISION, ALLOCATABLE               :: xf(:),yf(:),zf(:)
CHARACTER(LEN=10)                           :: cnumline,CNR,time
CHARACTER(LEN=20)                           :: string
!***********************************************************************
! HDF5
INTEGER(HID_T) :: file_id1,file_id2       ! File identifier
INTEGER(HID_T) :: dset_id1,dset_id2,dset_id3,dset_id4,dset_id5,dset_id6 ! Dataset identifier
INTEGER :: error1,error
INTEGER(HSIZE_T), DIMENSION(3) :: data_dims
INTEGER(HSIZE_T), DIMENSION(1) :: data_dims_grid

!***********************************************************************
! Define starting points on the BH surface
! These spherical coordinates are define with respect to the *MAGNETIC AXIS*
! and *NOT* to the STAR ROTATION AXIS.
!***********************************************************************

r0t=[1.1]
th0t=[0.15*pi,0.30*pi,0.40*pi,0.60*pi,0.70*pi,0.85*pi]
!th0t=[0.15*pi,0.30*pi]
!ph0t=[0d0,0.5d0*pi,pi,1.5d0*pi]
ph0t=[0d0,0.25d0*pi,0.5d0*pi,0.75d0*pi,pi,1.25d0*pi,1.5d0*pi,1.75d0*pi]

!***********************************************************************
! IMPORT the data !
!***********************************************************************


CALL h5open_f(error1)
CALL h5fopen_f('./data/grid_xyz.h5',H5F_ACC_RDWR_F,file_id1,error1)

data_dims_grid(1) = NX

CALL h5dopen_f(file_id1,'xpos',dset_id1,error1)
CALL h5dread_f(dset_id1,H5T_NATIVE_DOUBLE,x,data_dims_grid,error1)

data_dims_grid(1) = NY

CALL h5dopen_f(file_id1,'ypos',dset_id2,error1)
CALL h5dread_f(dset_id2,H5T_NATIVE_DOUBLE,y,data_dims_grid,error1)

data_dims_grid(1) = NZ

CALL h5dopen_f(file_id1,'zpos',dset_id3,error1)
CALL h5dread_f(dset_id3,H5T_NATIVE_DOUBLE,z,data_dims_grid,error1)

CALL h5dclose_f(dset_id1,error1)
CALL h5dclose_f(dset_id2,error1)
CALL h5dclose_f(dset_id3,error1)
CALL h5fclose_f(file_id1,error1)
CALL h5close_f(error1)

data_dims(1) = NX
data_dims(2) = NY
data_dims(3) = NZ


DO it0=tmin,tmax

WRITE(time,'(i5.5)') int(it0)
time=adjustl(time)
PRINT*,time

string='./data/Bxyz_' // trim(time) // '.h5'

! H5 data loading of Bru, Bthu, Bphu

CALL h5open_f(error)
CALL h5fopen_f(string,H5F_ACC_RDWR_F,file_id2,error)

CALL h5dopen_f(file_id2,'Bx',dset_id4,error)
CALL h5dread_f(dset_id4,H5T_NATIVE_DOUBLE,Bx,data_dims,error)

CALL h5dopen_f(file_id2,'By',dset_id5,error)
CALL h5dread_f(dset_id5,H5T_NATIVE_DOUBLE,By,data_dims,error)

CALL h5dopen_f(file_id2,'Bz',dset_id6,error)
CALL h5dread_f(dset_id6,H5T_NATIVE_DOUBLE,Bz,data_dims,error)

CALL h5dclose_f(dset_id4,error)
CALL h5dclose_f(dset_id5,error)
CALL h5dclose_f(dset_id6,error)
CALL h5fclose_f(file_id2,error)
CALL h5close_f(error)

!***********************************************************************
! COMPUTATIONS FIELD LINES
!***********************************************************************

! Length element in the Euler integration set at the grid spacing in r
ds=x(2)-x(1)

numline=0

!***********************************************************************
! Loop over all starting points
!***********************************************************************

DO irt=1,SIZE(r0t)
DO itht=1,SIZE(th0t)
DO ipht=1,SIZE(ph0t)

numline=numline+1

!***********************************************************************
! INITIALIZATION
!***********************************************************************

!***********************************************************************
! BEGIN transformation back to the Cartesian coordinates with respect
! to the ROTATION axis where the fields are defined.
!***********************************************************************

x1=r0t(irt)*sin(th0t(itht))*cos(ph0t(ipht))
y1=r0t(irt)*sin(th0t(itht))*sin(ph0t(ipht))
z1=r0t(irt)*cos(th0t(itht))


xt(1)=x1
yt(1)=y1
zt(1)=z1

!***********************************************************************
! Magnetic field components transformed in Cartesian coordinates
!***********************************************************************

 Bxt=TRILINEAR_BFIELD(Bx,x,y,z,x1,y1,z1,NX,NY,NZ)
 Byt=TRILINEAR_BFIELD(By,x,y,z,x1,y1,z1,NX,NY,NZ)
 Bzt=TRILINEAR_BFIELD(Bz,x,y,z,x1,y1,z1,NX,NY,NZ)

! Total magnetic field strength
Bt=sqrt(Bxt*Bxt+Byt*Byt+Bzt*Bzt)

 co=Bzt/Bt
 sico=Bxt/Bt
 sisi=Byt/Bt

 !***********************************************************************
 ! Euler integration along the field line
 !***********************************************************************
 
 xt(2)=xt(1)+ds*sico
 yt(2)=yt(1)+ds*sisi
 zt(2)=zt(1)+ds*co

 rt=sqrt(xt(2)*xt(2)+yt(2)*yt(2)+zt(2)*zt(2))


 x2 = xt(2)
 y2 = yt(2)
 z2 = zt(2)
 
it=2

!***********************************************************************
! Integrate until one reach back the star or the outer boundary
!***********************************************************************

DO WHILE ((rt.LT.rpml).AND.(rt.GT.rmin).AND.(Bt.GT.(0.0)).AND.(it.LT.NT))

 ! Interpolation of the fields to the field line location
 Bxt=TRILINEAR_BFIELD(Bx,x,y,z,x2,y2,z2,NX,NY,NZ)
 Byt=TRILINEAR_BFIELD(By,x,y,z,x2,y2,z2,NX,NY,NZ)
 Bzt=TRILINEAR_BFIELD(Bz,x,y,z,x2,y2,z2,NX,NY,NZ)
 
 Bt=sqrt(Bxt*Bxt+Byt*Byt+Bzt*Bzt)

 IF (Bt.GT.(0.0)) THEN

 co=Bzt/Bt
 sico=Bxt/Bt
 sisi=Byt/Bt

 IF (z1.LT.(0.0)) THEN
 k1x=-sico
 k1y=-sisi
 k1z=-co
 ELSE
 k1x=sico
 k1y=sisi
 k1z=co
 END IF
 
 !====================================================================
 ! Calcul k2
 !====================================================================
  
 x2=xt(it)+0.5*ds*k1x
 y2=yt(it)+0.5*ds*k1y
 z2=zt(it)+0.5*ds*k1z
 
 rt=sqrt(x2*x2+y2*y2+z2*z2)
 tht=acos(z2/rt)
 
  ! Interpolation of the fields to the field line location
 Bxt=TRILINEAR_BFIELD(Bx,x,y,z,x2,y2,z2,NX,NY,NZ)
 Byt=TRILINEAR_BFIELD(By,x,y,z,x2,y2,z2,NX,NY,NZ)
 Bzt=TRILINEAR_BFIELD(Bz,x,y,z,x2,y2,z2,NX,NY,NZ)
 
 Bt=sqrt(Bxt*Bxt+Byt*Byt+Bzt*Bzt)
 
 IF (Bt.GT.(0.0)) THEN
 
 co=Bzt/Bt
 sico=Bxt/Bt
 sisi=Byt/Bt

 IF (z1.LT.(0.0)) THEN
 k2x=-sico
 k2y=-sisi
 k2z=-co
 ELSE
 k2x=sico
 k2y=sisi
 k2z=co
 END IF
 
 !====================================================================
 ! Calcul k3
 !====================================================================
  
 x2=xt(it)+0.5*ds*k2x
 y2=yt(it)+0.5*ds*k2y
 z2=zt(it)+0.5*ds*k2z

 rt=sqrt(x2*x2+y2*y2+z2*z2)
 tht=acos(z2/rt)

  ! Interpolation of the fields to the field line location
 Bxt=TRILINEAR_BFIELD(Bx,x,y,z,x2,y2,z2,NX,NY,NZ)
 Byt=TRILINEAR_BFIELD(By,x,y,z,x2,y2,z2,NX,NY,NZ)
 Bzt=TRILINEAR_BFIELD(Bz,x,y,z,x2,y2,z2,NX,NY,NZ)
 
 Bt=sqrt(Bxt*Bxt+Byt*Byt+Bzt*Bzt)
 
 IF (Bt.GT.(0.0)) THEN
 
 co=Bzt/Bt
 sico=Bxt/Bt
 sisi=Byt/Bt

 IF (z1.LT.(0.0)) THEN
 k3x=-sico
 k3y=-sisi
 k3z=-co
 ELSE
 k3x=sico
 k3y=sisi
 k3z=co
 END IF
 
 !====================================================================
 ! Calcul k4
 !====================================================================
 
 x2=xt(it)+ds*k3x
 y2=yt(it)+ds*k3y
 z2=zt(it)+ds*k3z

 rt=sqrt(x2*x2+y2*y2+z2*z2)
 tht=acos(z2/rt)
 
  ! Interpolation of the fields to the field line location
 Bxt=TRILINEAR_BFIELD(Bx,x,y,z,x2,y2,z2,NX,NY,NZ)
 Byt=TRILINEAR_BFIELD(By,x,y,z,x2,y2,z2,NX,NY,NZ)
 Bzt=TRILINEAR_BFIELD(Bz,x,y,z,x2,y2,z2,NX,NY,NZ)
  
 Bt=sqrt(Bxt*Bxt+Byt*Byt+Bzt*Bzt)
 
 IF (Bt.GT.(0.0)) THEN
 
 co=Bzt/Bt
 sico=Bxt/Bt
 sisi=Byt/Bt

 IF (z1.LT.(0.0)) THEN
 k4x=-sico
 k4y=-sisi
 k4z=-co
 ELSE
 k4x=sico
 k4y=sisi
 k4z=co
 END IF
 
 !====================================================================
 ! Calcul x^n+1
 !====================================================================
 
 xt(it+1)=xt(it)+ds/6.*(k1x+2.*k2x+2.*k3x+k4x)
 yt(it+1)=yt(it)+ds/6.*(k1y+2.*k2y+2.*k3y+k4y)
 zt(it+1)=zt(it)+ds/6.*(k1z+2.*k2z+2.*k3z+k4z)
 
 it=it+1
 
 rt=sqrt(xt(it)*xt(it)+yt(it)*yt(it)+zt(it)*zt(it))
 tht=acos(zt(it)/rt)
 
 END IF
 END IF
 END IF
 END IF

END DO

! Final number of elements for the field line
NTF=it

!***********************************************************************
ALLOCATE(xf(1:NTF),yf(1:NTF),zf(1:NTF))

!***********************************************************************
! Transfer data to the array written to disk
!***********************************************************************

DO i=1,NTF
xf(i)=xt(i)
yf(i)=yt(i)
zf(i)=zt(i)
ENDDO

!***********************************************************************
! Writing to disk field line coordinates
!***********************************************************************

! Convert the integer it into a string cit
 WRITE(cnumline,'(I10)') numline
 cnumline=adjustl(cnumline)

!OPEN(9,FILE='./data_lines/line_' // trim(cnumline) // '_'// trim(time) // '.csv')

OPEN(9,FILE='./data_lines/lines_'// trim(time) // '.csv',POSITION='APPEND')
WRITE(9,*) 128.1d0,',',128.2d0,',',128.3d0
DO it=1,NTF
WRITE(9,*) xf(it)/rmax*128.+128.,',',yf(it)/rmax*128.+128.,',',zf(it)/rmax*128.+128.
ENDDO
CLOSE(9)

DEALLOCATE(xf,yf,zf)

ENDDO
ENDDO
ENDDO

ENDDO

END PROGRAM LOOP_FIELD_LINES

!***********************************************************************
! Function TRILINEAR_BFIELD: trilinear interpolation of the B field
! in 3D Cartesian coordinates.
!***********************************************************************

FUNCTION TRILINEAR_BFIELD(field,xi,yi,zi,xf,yf,zf,NX,NY,NZ)

IMPLICIT NONE

!!! INPUT/OUTPUT PARAMETERS !!!
INTEGER                                       :: NX,NY,NZ
DOUBLE PRECISION, DIMENSION(1:NX,1:NY,1:NZ) :: field
DOUBLE PRECISION, DIMENSION(1:NX)             :: xi
DOUBLE PRECISION, DIMENSION(1:NY)             :: yi
DOUBLE PRECISION, DIMENSION(1:NZ)             :: zi
DOUBLE PRECISION                              :: xf,yf,zf
DOUBLE PRECISION                              :: trilinear_bfield
INTEGER, DIMENSION(1)                         :: mini

!!! Intermediate variables !!!
DOUBLE PRECISION :: xp,yq,zr,f000,f100,f010,f001,f110,f101,f011,f111

!!! Loop indexes !!!
INTEGER          :: i,j,l
!***********************************************************************

i=FLOOR((xf-xi(1))/((xi(NX)-xi(1))/(NX-1)))+1
j=FLOOR((yf-yi(1))/((yi(NY)-yi(1))/(NY-1)))+1
l=FLOOR((zf-zi(1))/((zi(NZ)-zi(1))/(NZ-1)))+1

IF (i.EQ.NX) THEN
i=i-1
END IF

IF (j.EQ.NY) THEN
j=j-1
END IF

IF (l.EQ.NZ) THEN
l=l-1
END IF

xp=(xf-xi(i))/(xi(i+1)-xi(i))
yq=(yf-yi(j))/(yi(j+1)-yi(j))
zr=(zf-zi(l))/(zi(l+1)-zi(l))

!***********************************************************************

f000=field(i,j,l)
f100=field(i+1,j,l)
f010=field(i,j+1,l)
f001=field(i,j,l+1)
f110=field(i+1,j+1,l)
f101=field(i+1,j,l+1)
f011=field(i,j+1,l+1)
f111=field(i+1,j+1,l+1)

trilinear_bfield=f000*(1.0-xp)*(1.0-yq)*(1.0-zr)+f100*xp*(1.0-yq)*(1.0-zr)+&
f010*(1.0-xp)*yq*(1.0-zr)+f001*(1.0-xp)*(1.0-yq)*zr+f110*xp*yq*(1.0-zr)+&
f101*xp*(1.0-yq)*zr+f011*(1.0-xp)*yq*zr+f111*xp*yq*zr

!***********************************************************************

END FUNCTION TRILINEAR_BFIELD
