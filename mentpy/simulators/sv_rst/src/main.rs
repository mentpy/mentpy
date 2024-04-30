fn main() {
    println!("Hello, world!");

    let mut jacobian = vec![0.0; 14*1];
    let res = drun(&angles, &jacobian, 1.0);
    println!("{:?}", res);

    // for i in 0..10 {
    //     let mut dout = vec![0.0; 14];
    //     dout[i] = 1.0;
    //     drun(&angles, &mut dangles, &mut out, &mut dout);
    //     for i in 0..14 {
    //         jacobian[i*10 + i] = dangles[i];
    //     }
    // }
}

#[autodiff(drun, Reverse, Duplicated, Active)]
fn sum_logistic(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += 1.0 / (1.0 + (-x[i]).exp());
    }
    return sum;
}

//#[autodiff(drun, Reverse, Duplicated, Active)]
fn run(angles: &[f64]) -> f64 {
    // let (out, out2) = run(angles);
    // let res = userfn(out2);
    // return res;
    todo!();
}



