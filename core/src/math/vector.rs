pub trait Vector<T>
where
    Self: Copy + Clone + Default,
    T: Copy + Clone + Default,
{
    fn dot(self, other: Self) -> T;
    fn length(self) -> T;
    fn length_sq(self) -> T;
    fn length_inv(self) -> T;
    fn distance(self, other: Self) -> T;
    fn normalize(self) -> Self;
}
pub trait Vector2<T>: Vector<T>
where
    Self: Vector<T>,
    T: Copy + Clone + Default,
{
    fn new(x: T, y: T) -> Self;
    fn x(self) -> T;
    fn y(self) -> T;
    fn xy(self) -> (T, T);
    fn yx(self) -> (T, T);
    fn xx(self) -> (T, T);
    fn yy(self) -> (T, T);
}
