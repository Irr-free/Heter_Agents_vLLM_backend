#define _GNU_SOURCE

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t keep_running = 1;

typedef struct {
  int cpu;
  unsigned long long iterations;
  double checksum;
} worker_arg_t;

static void handle_signal(int sig) {
  (void)sig;
  keep_running = 0;
}

static void *worker_main(void *arg) {
  worker_arg_t *worker = (worker_arg_t *)arg;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(worker->cpu, &set);
  int rc = pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
  if (rc != 0) {
    fprintf(stderr, "pthread_setaffinity_np(cpu=%d) failed: %s\n", worker->cpu,
            strerror(rc));
  }

  double a = 1.0000001 + (double)(worker->cpu % 17) * 0.000001;
  double b = 0.9999997 + (double)(worker->cpu % 13) * 0.000001;
  double c = 1.0000003 + (double)(worker->cpu % 11) * 0.000001;
  double d = 0.9999991 + (double)(worker->cpu % 7) * 0.000001;
  unsigned long long iter = 0;

  while (keep_running) {
    for (int i = 0; i < 100000; ++i) {
      a = a * b + c;
      b = b * c + d;
      c = c * d + a;
      d = d * a + b;
      if (a > 1e100 || b > 1e100 || c > 1e100 || d > 1e100) {
        a = fmod(a, 1.0) + 1.0000001;
        b = fmod(b, 1.0) + 0.9999997;
        c = fmod(c, 1.0) + 1.0000003;
        d = fmod(d, 1.0) + 0.9999991;
      }
    }
    iter += 100000ULL;
  }

  worker->iterations = iter;
  worker->checksum = a + b + c + d;
  return NULL;
}

static int parse_cpu_list(const char *text, int **out_cpus) {
  if (text == NULL || text[0] == '\0') {
    *out_cpus = NULL;
    return 0;
  }

  char *copy = strdup(text);
  if (copy == NULL) {
    return -1;
  }

  int capacity = 64;
  int count = 0;
  int *cpus = calloc((size_t)capacity, sizeof(int));
  if (cpus == NULL) {
    free(copy);
    return -1;
  }

  char *save = NULL;
  for (char *token = strtok_r(copy, ",", &save); token != NULL;
       token = strtok_r(NULL, ",", &save)) {
    char *dash = strchr(token, '-');
    if (dash != NULL) {
      *dash = '\0';
      int start = atoi(token);
      int end = atoi(dash + 1);
      if (end < start) {
        int tmp = start;
        start = end;
        end = tmp;
      }
      for (int cpu = start; cpu <= end; ++cpu) {
        if (count == capacity) {
          capacity *= 2;
          int *new_cpus = realloc(cpus, (size_t)capacity * sizeof(int));
          if (new_cpus == NULL) {
            free(cpus);
            free(copy);
            return -1;
          }
          cpus = new_cpus;
        }
        cpus[count++] = cpu;
      }
    } else {
      if (count == capacity) {
        capacity *= 2;
        int *new_cpus = realloc(cpus, (size_t)capacity * sizeof(int));
        if (new_cpus == NULL) {
          free(cpus);
          free(copy);
          return -1;
        }
        cpus = new_cpus;
      }
      cpus[count++] = atoi(token);
    }
  }

  free(copy);
  *out_cpus = cpus;
  return count;
}

static double monotonic_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

int main(int argc, char **argv) {
  if (argc < 3 || strcmp(argv[1], "--cpus") != 0) {
    fprintf(stderr, "usage: %s --cpus <cpu-list>\n", argv[0]);
    return 2;
  }

  int *cpus = NULL;
  int num_cpus = parse_cpu_list(argv[2], &cpus);
  if (num_cpus < 0) {
    fprintf(stderr, "failed to parse cpu list\n");
    return 2;
  }
  if (num_cpus == 0) {
    fprintf(stderr, "empty cpu list; exiting\n");
    return 0;
  }

  signal(SIGINT, handle_signal);
  signal(SIGTERM, handle_signal);

  pthread_t *threads = calloc((size_t)num_cpus, sizeof(pthread_t));
  worker_arg_t *args = calloc((size_t)num_cpus, sizeof(worker_arg_t));
  if (threads == NULL || args == NULL) {
    perror("calloc");
    free(cpus);
    free(threads);
    free(args);
    return 1;
  }

  for (int i = 0; i < num_cpus; ++i) {
    args[i].cpu = cpus[i];
    int rc = pthread_create(&threads[i], NULL, worker_main, &args[i]);
    if (rc != 0) {
      fprintf(stderr, "pthread_create(cpu=%d) failed: %s\n", cpus[i],
              strerror(rc));
      keep_running = 0;
      num_cpus = i;
      break;
    }
  }

  double start = monotonic_seconds();
  fprintf(stderr, "core_burner started workers=%d cpus=%s\n", num_cpus, argv[2]);
  fflush(stderr);

  while (keep_running) {
    sleep(1);
  }

  for (int i = 0; i < num_cpus; ++i) {
    pthread_join(threads[i], NULL);
  }

  double elapsed = monotonic_seconds() - start;
  unsigned long long total_iter = 0;
  double checksum = 0.0;
  for (int i = 0; i < num_cpus; ++i) {
    total_iter += args[i].iterations;
    checksum += args[i].checksum;
  }
  fprintf(stderr,
          "core_burner stopped workers=%d elapsed_s=%.3f iterations=%llu "
          "checksum=%.6e\n",
          num_cpus, elapsed, total_iter, checksum);

  free(cpus);
  free(threads);
  free(args);
  return 0;
}
