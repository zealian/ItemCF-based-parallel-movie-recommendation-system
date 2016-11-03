//This program is an implementation of Item-based Collaborative Filtering. 
//The aim of this program is to recommend movie for users.
//Data of this program is the MovieLens 1M dataset from http://grouplens.org/datasets/movielens/
//Author: Anni Piao, Hanning Su
//StudentID: 734514, 741018

#include <cstdio>
#include <fstream>
#include <cmath>
#include <math.h>
#include <omp.h>
#include <cstdlib>
#include <memory.h>
#include <mpi.h>
using namespace std;

#define user_num 6050
#define movie_num 4000
#define rating_num 1000300
#define nElems(x)  (sizeof(x) / sizeof((x)[0]))

struct movie{
	int id;
	char name[128];
	char tags[128];
};

struct rating{
	int user_id;
	int movie_id;
	int grade;
	int time;
};

struct ranking{
    int movie_id;
    double score;
};

struct movie movies_info[movie_num];
struct rating ratings_info[rating_num];
int movieID2Line[movie_num];
int movieNuser[user_num][movie_num]; //user_movie matrix, value is score
double **similarity; //the similarity between movies
int real_movie_num = 0;
int real_user_num = 0;
int real_rating_num = 0;

int load_movies(){
    fstream movies;
    char str[256];

	movies.open("movies.dat",ios::in|ios::binary);
	if (!movies.is_open()){
		printf("file open failed");
		return -1;
	}
    int i = 0;
    while (movies.peek()!=EOF){ //didn't reach the end
        movies.getline(str,255); //get one line from the file
    	sscanf(str,"%d::%[^::]::%s",&movies_info[i].id,&movies_info[i].name,&movies_info[i].tags);
        movieID2Line[movies_info[i].id]=i;
        ++i;
    }
    real_movie_num = i;
    movies.close();
    return 1;
}

int load_ratings(){
    fstream ratings;
    char str[256];

    ratings.open("ratings.dat",ios::in|ios::binary);
    if (!ratings.is_open()){
        printf("file open failed");
        return -1;
    }

    int lastID;
    int i = 0;
    while (ratings.peek()!=EOF){ //didn't reach the end
        ratings.getline(str,255); //get one line from the file
        sscanf(str,"%d::%d::%d::%d",&ratings_info[i].user_id,
            &ratings_info[i].movie_id,&ratings_info[i].grade,&ratings_info[i].time);
        if(real_user_num==0){
            ++real_user_num;
            lastID = ratings_info[i].user_id;
            continue;
        }
        if(ratings_info[i].user_id!=lastID){
            lastID = ratings_info[i].user_id;
            ++real_user_num;
        }
        ++i;
    }
    real_rating_num=i;
    ratings.close();
    return 1;
}

void update_movieNuser(){
    for(int i = 0; i < real_rating_num; ++i){
        movieNuser[ratings_info[i].user_id-1][movieID2Line[ratings_info[i].movie_id]] = ratings_info[i].grade;
    }
}

//Create an 2d array with contiguous addresses
double **createArray(int n, int m) {
    double *data = (double *)calloc(n*m,sizeof(double));
    double **array = (double **)calloc(n, sizeof(double *));
    for (int i=0; i<n; i++)
        array[i] = &(data[i*m]);

    return array;
}

void freeArray(double **array) {
    free(array[0]);
    free(array);
}

//Calculate similarity of two 1d array by Pearson correlation 
double cal_similarity(int lower,int i, int j){
    int num, sum_x, sum_y, sum_x2, sum_y2, sum_xy;
    double nu, de, sim;
    int global_i,global_j;
    num = 0, sum_x = 0, sum_y = 0;
    sum_x2 = 0, sum_y2= 0, sum_xy = 0;
    global_j = lower + j;
    global_i = i;
    for(int u = 0; u < real_user_num; ++u){
        if(movieNuser[u][global_i]>0&&movieNuser[u][global_j]>0){
            ++num;
            sum_x+=movieNuser[u][global_i];
            sum_y+=movieNuser[u][global_j];
            sum_x2+=(movieNuser[u][global_i]*movieNuser[u][global_i]);
            sum_y2+=(movieNuser[u][global_j]*movieNuser[u][global_j]);
            sum_xy+=(movieNuser[u][global_i]*movieNuser[u][global_j]);
            }
        }

    if(num == 0 || num == 1){
        return 0.0000;
    }

    de = sqrt((sum_x2 - sum_x*sum_x/num)*(sum_y2 - sum_y*sum_y/num));

    if(de == 0){
        return 0.0000;
    }

    nu = sum_xy-(sum_x*sum_y/num);
    sim =nu / de;

    return sim;
}

//Calculate similarity of all movies. MPI Allgatherv is applied to achieve higher performance.
void movie_similarity(){
    int numtasks,rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int size = real_movie_num;
    int row,col;
    int lower, upper;
    int pre_chunk,cur_chunk;
    double **local_sim;
    int *displs = new int[numtasks];
    int *chunkList = new int[numtasks];

    cur_chunk = size / numtasks;
    lower = rank*cur_chunk;
    if(rank == numtasks -1){
        upper = size;
        cur_chunk = upper - lower;
    }else{
        upper = (rank+1)*cur_chunk;
    }

    int chunk_size = size/numtasks;
    for(int i = 0; i < numtasks;++i){
        displs[i] = i*chunk_size;
        if(i!=numtasks-1){
            chunkList[i] = chunk_size;
        }else{
            chunkList[i] = size - i*chunk_size;
        }
    }

    if(rank==0){
        pre_chunk = 0;
    }else{
        pre_chunk = chunk_size;
    }

    row = size;
    col = cur_chunk;
    local_sim=createArray(row,col);

    similarity=createArray(size,size);

    if(numtasks!=1){
        
        if(rank!=0){
            //Rectangular
            for(int i = 0; i < pre_chunk; ++i){
                for(int j = 0; j < cur_chunk; ++j){
                    local_sim[i][j] = cal_similarity(lower,i,j);
                }
            }        
        }      

        //Triangle
        for(int i = pre_chunk; i < pre_chunk + cur_chunk; ++i){
            for(int j = i - pre_chunk + 1; j < cur_chunk; ++j){
                local_sim[i][j] = cal_similarity(lower,i,j);
            }
        }

        //MPI type for local matrix
        MPI_Datatype a_col_type, new_a_col_type;
        MPI_Type_vector(row, 1, col, MPI_DOUBLE, &a_col_type);
        MPI_Type_commit(&a_col_type);

        MPI_Type_create_resized(a_col_type, 0, 1*sizeof(double), &new_a_col_type);
        MPI_Type_commit(&new_a_col_type);

        //MPI type for global matrix
        MPI_Datatype b_col_type, new_b_col_type;
        MPI_Type_vector(size, 1, size, MPI_DOUBLE, &b_col_type);
        MPI_Type_commit(&b_col_type);

        MPI_Type_create_resized(b_col_type, 0, 1*sizeof(double), &new_b_col_type);
        MPI_Type_commit(&new_b_col_type);

        MPI_Allgatherv(local_sim[0], chunkList[rank], new_a_col_type,
                similarity[0], chunkList, displs, new_b_col_type,
                MPI_COMM_WORLD);

    }else{
        for(int i = pre_chunk; i < pre_chunk + cur_chunk; ++i){
            for(int j = i - pre_chunk + 1; j < cur_chunk; ++j){
                similarity[i][j] = cal_similarity(lower,i,j);
            }
        }
    }

        for(int i = 0; i < size; ++i){
            for(int j = 0; j < i; ++j){
                similarity[i][j]=similarity[j][i];
            }
        }
}

void quicksort(struct ranking* ranking_info,int first, int last,int top_num){
    int pivot,j,i;
    struct ranking temp;

    while(first<last){
        pivot=first;
        i=first;
        j=last;

        //Partitioning
        while(i<j){
            while(ranking_info[i].score>=ranking_info[pivot].score&&i<last)
                ++i;
            while(ranking_info[j].score<ranking_info[pivot].score)
                --j;
            if(i<j){
                temp=ranking_info[i];
                ranking_info[i]=ranking_info[j];
                ranking_info[j]=temp;
            }
        }
        temp=ranking_info[pivot];
        ranking_info[pivot]=ranking_info[j];
        ranking_info[j]=temp;

        if(top_num<=j){
            last = j -1;
        }else if(j - first > last - j){
            quicksort(ranking_info,j+1,last,top_num);
            last = j - 1;
        }else{
            quicksort(ranking_info,first,j-1,top_num);
            first = j + 1;
        }
    }
}

void partition(struct ranking* ranking_info,int first, int last,int top_num,int offset){
    int pivot,j,i;
    struct ranking temp;

    while(first<last){
        pivot=first;
        i=first;
        j=last;

        //Partitioning
        while(i<j){
            while(ranking_info[i].score>=ranking_info[pivot].score&&i<last)
                ++i;
            while(ranking_info[j].score<ranking_info[pivot].score)
                --j;
            if(i<j){
                temp=ranking_info[i];
                ranking_info[i]=ranking_info[j];
                ranking_info[j]=temp;
            }
        }
        temp=ranking_info[pivot];
        ranking_info[pivot]=ranking_info[j];
        ranking_info[j]=temp;

        if(top_num==j-offset+1){
            break;
        }else if(top_num<j-offset+1){
            last = j -1;
        }else{
            if(j - first > last - j){
                partition(ranking_info,j+1,last,top_num,offset);
                last = j - 1;
            }else{
                partition(ranking_info,first,j-1,top_num,offset);
                first = j + 1;
            }
        } 
    }
}

int* sort(int top_num, struct ranking* ranking_info){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int num = omp_get_max_threads();
    struct ranking *result = new ranking[num*top_num];
    
    if(real_movie_num/num<top_num){
        num = (int) (real_movie_num / top_num); 
    }

    #pragma omp parallel num_threads(num)
    {
        int id = omp_get_thread_num();
        int lower,upper;
        int chunk = real_movie_num / num;
        lower = id * chunk;
        if(id != num -1){
            upper = (id+1) * chunk - 1;
        }else{
            upper = real_movie_num-1;
        }
        partition(ranking_info,lower,upper,top_num,lower);
        for(int i=0;i<top_num;++i){
            result[i+id*top_num] = ranking_info[lower+i];
        }
    }
    quicksort(result,0,num*top_num-1,top_num);
    int *r = new int[top_num];
    for(int i=0;i<top_num;++i){
        r[i] = result[i].movie_id;
    }
    return r;
}

void score(int uid, struct ranking* ranking_info){
    for(int i=0;i<real_movie_num;++i){
        ranking_info[i].movie_id = movies_info[i].id;
        ranking_info[i].score = 0;
        for(int j=0;j<real_movie_num;++j){
            ranking_info[i].score+=(movieNuser[uid-1][j]*similarity[i][j]);
        }
    }
}

int* getTop(int user, int top_num){
    struct ranking ranking_info[movie_num];
    score(user,ranking_info);
    return sort(top_num, ranking_info);
}

void save_sim_matrix(){
    FILE *f = fopen("test.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for(int i = 0; i < real_movie_num; ++i){
        for(int j = 0; j < real_movie_num; ++j){
            fprintf(f, "%.4f ", similarity[i][j]);
            if(j == real_movie_num-1){
                fprintf(f, "%.4f\n", similarity[i][j]);
            }
        }
    }
    fclose(f);
}

void load_sim_matrix(){
    FILE* file = fopen ("test.txt", "r");
    similarity=createArray(real_movie_num,real_movie_num);
    for(int i = 0; i < real_movie_num; ++i){
        for(int j = 0; j < real_movie_num; ++j){
            fscanf(file,"%lf",&similarity[i][j]);
        }
    }
    fclose (file);
}

void output(int user,int* result,int top_num){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("=====================================================================\n");
    printf("                     Recommendation list for user %d\n",user);
    printf("=====================================================================\n");
    printf(" Rank | Movie Name\n");

    for(int i = 0; i < top_num; ++i){
        printf("------|--------------------------------------------------------------\n");
        printf(" %4d | %s\n",i+1,movies_info[movieID2Line[result[i]]].name);
    }
    printf("=====================================================================\n");
}

//Command line arguments: mode top_k uid1 uid2 [...]
int main(int argc, char *argv[]){  
    MPI_Init(&argc,&argv);
    omp_set_num_threads(8);
    int numtasks,rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int *result;
    int user,top_num,mode;
    int uid_num;

    load_movies();
    load_ratings();
    update_movieNuser();

    sscanf(argv[1],"%d",&mode);
    if(mode==0){
        printf("Mode 0.\n");
        movie_similarity();
        save_sim_matrix();
    }else{
        printf("Mode 1.\n");
        load_sim_matrix();
    }

    sscanf(argv[2],"%d",&top_num);
    uid_num = argc-3;
    
    for(int i=0;i<uid_num;i++){
        if(i%numtasks!=rank)
            continue;
        sscanf(argv[i+3],"%d",&user);
        result=getTop(user, top_num);
        output(user,result,top_num);
    }

    MPI_Finalize(); 
    return 0;
}