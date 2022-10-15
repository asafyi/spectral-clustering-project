int fit(double** data, double** cent,int k,int dim,int vec_count);
void to_weighted(double ** weight, double ** vectors,int dim,int vec_counter);
void to_diagonal(double ** diagonal,double ** weight,int vec_counter);
void to_lnorm(double ** lnorm,double ** diagonal,double ** weight,int vec_counter);
int to_jacobian(double ** jacob,double ** lnorm,int vec_counter);
int eigengap(double ** jacobi,double ** T,int vec_counter,int k);
