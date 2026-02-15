// Gmsh project created on Mon Dec 22 16:18:53 2025
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0.4, 0, 0, 1.0};
//+
Point(3) = {1, 0, 0, 1.0};
//+
Point(4) = {1, 0.4, 0, 1.0};
//+
Point(5) = {0.4, 0.4, 0, 1.0};
//+
Point(6) = {0.4, 1, 0, 1.0};
//+
Point(7) = {0, 1, 0, 1.0};
//+
Point(8) = {0, 0.4, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 8};
//+
Line(8) = {8, 1};
//+
Line(9) = {8, 5};
//+
Line(10) = {5, 2};
//+
Curve Loop(1) = {7, 9, 5, 6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {8, 1, -10, -9};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {4, 10, 2, 3};
//+
Plane Surface(3) = {3};
//+
Physical Surface("main", 11) = {1, 2, 3};
//+
Transfinite Curve {7, 5, 4, 2} = 60 Using Progression 1;
//+
Transfinite Curve {6, 9, 8, 10, 1, 3} = 40 Using Progression 1;
