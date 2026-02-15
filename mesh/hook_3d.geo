// Gmsh project created on Thu Jan 29 13:08:18 2026
//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0, 0, 0.4, 1.0};
//+
Point(3) = {0.4, 0, 0.4, 1.0};
//+
Point(4) = {0.4, 0, 0, 1.0};
//+
Point(5) = {1, 0, 0, 1.0};
//+
Point(6) = {1, 0, 0.4, 1.0};
//+
Point(7) = {1, 0.4, 0.4, 1.0};
//+
Point(8) = {1, 0.4, 0, 1.0};
//+
Point(9) = {0.4, 0.4, 0, 1.0};
//+
Point(10) = {0.4, 0.4, 0.4, 1.0};
//+
Point(11) = {0, 0.4, 0.4, 1.0};
//+
Point(12) = {0, 0.4, 0, 1.0};
//+
Point(13) = {0, 1, 0, 1.0};
//+
Point(14) = {0, 1, 0.4, 1.0};
//+
Point(15) = {0.4, 1, 0.4, 1.0};
//+
Point(16) = {0.4, 1, 0, 1.0};
//+
Line(1) = {2, 3};
//+
Line(2) = {3, 4};
//+
Line(3) = {4, 1};
//+
Line(4) = {1, 2};
//+
Line(5) = {3, 6};
//+
Line(6) = {6, 5};
//+
Line(7) = {5, 4};
//+
Line(8) = {3, 10};
//+
Line(9) = {10, 11};
//+
Line(10) = {11, 2};
//+
Line(11) = {11, 12};
//+
Line(12) = {12, 1};
//+
Line(13) = {12, 9};
//+
Line(14) = {9, 10};
//+
Line(15) = {9, 4};
//+
Line(16) = {10, 7};
//+
Line(17) = {7, 6};
//+
Line(18) = {7, 8};
//+
Line(19) = {8, 5};
//+
Line(20) = {8, 9};
//+
Line(21) = {9, 16};
//+
Line(22) = {16, 15};
//+
Line(23) = {15, 10};
//+
Line(24) = {15, 14};
//+
Line(25) = {14, 11};
//+
Line(26) = {12, 13};
//+
Line(27) = {13, 14};
//+
Line(28) = {13, 16};
//+
Curve Loop(1) = {9, 10, 1, 8};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {16, 17, -5, 8};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {20, 15, -7, -19};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {13, 15, 3, -12};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {18, 19, -6, -17};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {11, 12, 4, -10};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {3, 4, 1, 2};
//+
Plane Surface(7) = {7};
//+
Curve Loop(8) = {7, -2, 5, 6};
//+
Plane Surface(8) = {8};
//+
Curve Loop(9) = {14, 16, 18, 20};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {13, 14, 9, 11};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {23, -14, 21, 22};
//+
Plane Surface(11) = {11};
//+
Curve Loop(12) = {24, 25, -9, -23};
//+
Plane Surface(12) = {12};
//+
Curve Loop(13) = {27, 25, 11, 26};
//+
Plane Surface(13) = {13};
//+
Curve Loop(14) = {28, -21, -13, 26};
//+
Plane Surface(14) = {14};
//+
Curve Loop(15) = {24, -27, 28, 22};
//+
Plane Surface(15) = {15};
//+
Curve Loop(16) = {8, -14, 15, -2};
//+
Plane Surface(16) = {16};
//+
Surface Loop(1) = {12, 15, 13, 14, 11, 10};
//+
Volume(1) = {1};
//+
Surface Loop(2) = {1, 6, 4, 7, 10, 16};
//+
Volume(2) = {2};
//+
Surface Loop(3) = {2, 9, 5, 3, 8, 16};
//+
Volume(3) = {3};
//+
Physical Volume("main", 29) = {1, 2, 3};
//+
Transfinite Curve {25, 26, 21, 23, 20, 16, 5, 7} = 45 Using Progression 1;
//+
Transfinite Curve {24, 27, 28, 22, 9, 11, 13, 14, 12, 10, 4, 3, 1, 2, 15, 8, 17, 18, 19, 6} = 30 Using Progression 1;
