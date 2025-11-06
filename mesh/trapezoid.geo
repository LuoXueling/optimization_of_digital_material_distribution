// Gmsh project created on Fri Oct 17 14:44:06 2025
SetFactory("OpenCASCADE");
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {2, 0, 0, 1.0};
//+
Point(3) = {1.5, 1, 0, 1.0};
//+
Point(4) = {0.5, 1, 0, 1.0};
//+
Point(5) = {0.5-0.02, 1-0.04, 0, 1.0};
//+
Point(6) = {1.5+0.02, 1-0.04, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 6};
//+
Line(6) = {6, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {5, 1};

//+
Line(7) = {5, 6};
//+
Curve Loop(1) = {3, 4, 7, 6};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {5, 1, 2, -7};
//+
Plane Surface(2) = {2};
//+
Physical Curve("top", 8) = {3};
//+
Physical Curve("bottom", 9) = {1};
//+
Physical Surface("fix_design", 10) = {1};
//+
Physical Surface("main", 11) = {2};
//+
Transfinite Curve {3, 7, 1} = 50 Using Progression 1;
//+
Transfinite Curve {5, 2} = 46 Using Progression 1;
//+
Transfinite Curve {4, 6} = 3 Using Progression 1;
