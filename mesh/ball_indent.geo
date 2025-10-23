//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {1, 0, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {1, 1.2, 0, 1.0};
//+
Point(5) = {0.6, 1.2, 0, 1.0};
//+
Point(6) = {0.5, 1.1, 0, 1.0};
//+
Point(7) = {0.4, 1.2, 0, 1.0};
//+
Point(8) = {0, 1.2, 0, 1.0};
//+
Point(9) = {0, 1, 0, 1.0};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 5};
//+
Line(5) = {7, 8};
//+
Line(6) = {8, 9};
//+
Line(7) = {9, 1};
//+
Line(8) = {9, 3};
//+
Point(10) = {0.5, 1.2, 0, 1.0};
//+
Circle(9) = {7, 10, 5};
//+
Curve Loop(1) = {4, -9, 5, 6, 8, 3};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {7, 1, 2, -8};
//+
Plane Surface(2) = {2};
//+
Physical Surface("main", 10) = {2};
//+
Physical Surface("third_medium", 11) = {1};
//+
Physical Curve("bottom", 12) = {1};
//+
Physical Curve("ball", 13) = {9};
//+
Transfinite Curve {7, 1, 2, 8} = 40 Using Progression 1;
//+
Transfinite Curve {9} = 20 Using Progression 1;
//+
Transfinite Curve {5, 4} = 20 Using Progression 1;
//+
Transfinite Curve {6, 3} = 10 Using Progression 1;
