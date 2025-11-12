// Gmsh project created on Tue Nov  4 12:19:45 2025
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0.25, 0, 0, 1.0};
//+
Point(3) = {0.75, 0, 0, 1.0};
//+
Point(4) = {1.25, 0, 0, 1.0};
//+
Point(5) = {1.5, 0, 0, 1.0};
//+
Point(6) = {1.5, 0.75, 0, 1.0};
//+
Point(7) = {0, 0.75, 0, 1.0};
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
Line(7) = {7, 1};
//+
Circle(8) = {4, 3, 2};
//+
Physical Curve("semisphere-top", 12) = {8};
//+
Physical Curve("top", 9) = {6};
//+
Physical Curve("third_medium_bottom", 10) = {1, 4};
//+
Physical Curve("bottom", 11) = {2, 3};
//+
Curve Loop(1) = {6, 7, 1, -8, 4, 5};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {8, 2, 3};
//+
Plane Surface(2) = {2};
//+
Physical Surface("third_medium", 13) = {1};
//+
Physical Surface("main", 14) = {2};
//+
Transfinite Curve {7, 5} = 10 Using Progression 1;
//+
Transfinite Curve {6} = 20 Using Progression 1;
//+
Transfinite Curve {1, 4} = 5 Using Progression 1;
//+
Transfinite Curve {8} = 80 Using Progression 1;
//+
Transfinite Curve {2} = 50 Using Progression 1;
//+
Transfinite Curve {3} = 50 Using Progression 1;
