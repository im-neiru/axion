use core::arch::x86_64::*;
use std::mem::MaybeUninit;

use crate::math::{Vector, Vector2};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct FVector2(pub(crate) __m128);

impl Default for FVector2 {
    #[inline]
    fn default() -> Self {
        Self(unsafe { _mm_setzero_ps() })
    }
}

impl Vector2<f32> for FVector2 {
    #[inline]
    fn new(x: f32, y: f32) -> Self {
        Self(unsafe { _mm_set_ps(0.0, 0.0, y, x) })
    }

    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn x(self) -> f32 {
        unsafe {
            let mut x: f32 = MaybeUninit::uninit().assume_init();
            _mm_store_ss(&mut x, self.0);

            x
        }
    }

    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn y(self) -> f32 {
        unsafe {
            let mut y: f32 = MaybeUninit::uninit().assume_init();

            let v = _mm_shuffle_ps::<1>(self.0, self.0);
            _mm_store_ss(&mut y, v);

            y
        }
    }

    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn xy(self) -> (f32, f32) {
        unsafe {
            let mut x: f32 = MaybeUninit::uninit().assume_init();
            let mut y: f32 = MaybeUninit::uninit().assume_init();

            _mm_store_ss(&mut x, self.0);
            let v = _mm_shuffle_ps::<1>(self.0, self.0);
            _mm_store_ss(&mut y, v);

            (x, y)
        }
    }

    #[inline]
    fn yx(self) -> (f32, f32) {
        let xy = self.xy();
        (xy.1, xy.0)
    }

    #[inline]
    fn xx(self) -> (f32, f32) {
        let x = self.x();
        (x, x)
    }

    #[inline]
    fn yy(self) -> (f32, f32) {
        let y = self.y();
        (y, y)
    }
}

impl Vector<f32> for FVector2 {
    #[inline]
    #[allow(clippy::uninit_assumed_init)]
    #[allow(invalid_value)]
    fn dot(self, other: Self) -> f32 {
        unsafe {
            let mut dot_product: f32 = MaybeUninit::uninit().assume_init();
            let product = _mm_mul_ps(self.0, other.0);
            let sum = _mm_hadd_ps(product, product);

            _mm_store_ss(&mut dot_product, sum);

            dot_product
        }
    }

    #[inline]
    fn length(self) -> f32 {
        self.length_sq().sqrt()
    }

    #[inline]
    fn length_sq(self) -> f32 {
        self.dot(self)
    }

    #[inline]
    fn length_inv(self) -> f32 {
        self.length().recip()
    }
}

#[test]
fn test_fvector2_sse2() {
    use std::time::SystemTime;

    let a = FVector2::new(2.0, 3.0);
    let b = FVector2::new(4.0, 3.2);
    let ref_time = SystemTime::now();
    let dot = a.dot(b);

    let span_dot = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    let ref_time = SystemTime::now();
    let length = a.length();

    let span_length = SystemTime::now()
        .duration_since(ref_time)
        .unwrap()
        .as_nanos();

    assert!(dot == 17.6, "FVector2::dot | Wrong output");
    assert!(length == 3.6055512, "FVector2::length | Wrong output");

    println!(
        "FVector2::dot | {span_dot} ns | {:?} {:?} = {dot}",
        a.xy(),
        b.xy()
    );

    println!(
        "FVector2::length | {span_length} ns | {:?} = {length}",
        a.xy(),
    );
}
