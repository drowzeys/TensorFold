// Checks and benchmarks for libbolt on Metal.
//   bolt_test parity MODEL ORACLE_DIR PROMPT_LEN   teacher-forced KL and top-1 against fp32 reference logits:
//                                                  plain one-token steps, then prefill + 8-row verify windows
//   bolt_test gen MODEL IDS_FILE N DRAFT           greedy generation of N tokens (DRAFT 1: LiLiCorr drafts)
//   bolt_test prefill MODEL N                      prompt pass of N tokens
// IDS_FILE: a JSON list of prompt token ids.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

extern "C" {
struct Bolt;
Bolt* bolt_open(const char* dir, int context);
int bolt_prefill(Bolt*, const int*, int, int);
int bolt_forward(Bolt*, const int*, int, int, int);
void bolt_keep(Bolt*, int);
void bolt_push(Bolt*, unsigned);
void bolt_reset(Bolt*);
int bolt_propose(Bolt*, int, int*);
void bolt_candidates(Bolt*, int, int, float*, int*);
void bolt_logits_row(Bolt*, int, float*);
void bolt_prof_report(Bolt*);
}

static double now_ms() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
constexpr int V = 131072;

static std::vector<int> read_ids(const char* path) {
    std::vector<int> v;
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    long cur = 0; bool in = false;
    for (char c : s) {
        if (c >= '0' && c <= '9') { cur = cur * 10 + (c - '0'); in = true; }
        else { if (in) v.push_back((int)cur); cur = 0; in = false; }
    }
    return v;
}

// float32 .npy, rows of V.
static std::vector<float> read_npy(const char* path, size_t& rows) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    unsigned char hdr[10];
    fread(hdr, 1, 10, f);
    unsigned hlen = hdr[8] | (hdr[9] << 8);
    fseek(f, 10 + hlen, SEEK_SET);
    fseek(f, 0, SEEK_END);
    long end = ftell(f);
    fseek(f, 10 + hlen, SEEK_SET);
    rows = (end - 10 - hlen) / 4 / V;
    std::vector<float> d(rows * V);
    fread(d.data(), 4, d.size(), f);
    fclose(f);
    return d;
}

static void logsoftmax(const float* x, std::vector<double>& out) {
    double m = -1e300;
    for (int i = 0; i < V; i++) m = std::max(m, (double)x[i]);
    double s = 0;
    for (int i = 0; i < V; i++) s += exp(x[i] - m);
    double l = m + log(s);
    out.resize(V);
    for (int i = 0; i < V; i++) out[i] = x[i] - l;
}

struct Score { double kl = 0, klmax = 0; int top1 = 0, n = 0; };
static void score(Score& s, const float* ref, const float* got) {
    std::vector<double> a, b;
    logsoftmax(ref, a);
    logsoftmax(got, b);
    double kl = 0;
    int ia = 0, ib = 0;
    for (int i = 0; i < V; i++) {
        kl += exp(a[i]) * (a[i] - b[i]);
        if (a[i] > a[ia]) ia = i;
        if (b[i] > b[ib]) ib = i;
    }
    s.kl += kl;
    s.klmax = std::max(s.klmax, kl);
    s.top1 += ia == ib;
    s.n++;
}

static void report(const char* what, const Score& s) {
    printf("%-22s %4d positions  KL mean %.5f max %.5f  top-1 %d/%d (%.1f%%)\n", what, s.n, s.kl / s.n, s.klmax, s.top1, s.n, 100.0 * s.top1 / s.n);
    fflush(stdout);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: see the header of bolt_test.mm\n"); return 2; }
    std::string mode = argv[1];
    const char* ctx = getenv("BOLT_CONTEXT");
    Bolt* b = bolt_open(argv[2], ctx ? atoi(ctx) : 65536);
    if (!b) return 1;
    if (mode == "parity") {
        std::string dir = argv[3];
        int P = atoi(argv[4]);
        auto ids = read_ids((dir + "/ids.json").c_str());
        size_t rows;
        auto ref = read_npy((dir + "/logits.npy").c_str(), rows);
        const int n = (int)std::min(rows, ids.size()) - 1;
        std::vector<float> got(V);
        // 1. one token at a time (each a one-row step)
        Score s1;
        double t0 = now_ms();
        for (int i = 0; i < n; i++) {
            bolt_push(b, ids[i]);
            bolt_logits_row(b, 0, got.data());
            score(s1, &ref[(size_t)i * V], got.data());
        }
        report("one-token steps", s1);
        fprintf(stderr, "steps: %.1f ms a token\n", (now_ms() - t0) / n);
        // 2. prefill the prompt, then verify windows of 8 rows kept whole, and of 8 rows cut to 5
        bolt_reset(b);
        Score s2, s3;
        bolt_prefill(b, ids.data(), P, 1);
        bolt_logits_row(b, 0, got.data());
        score(s2, &ref[(size_t)(P - 1) * V], got.data());
        report("prefill last row", s2);
        int at = P;
        bool cut = false;
        while (at + 8 <= n) {
            bolt_forward(b, ids.data() + at, 8, 1, 1);
            const int keep = cut ? 5 : 8;
            for (int r = 0; r < keep; r++) {
                bolt_logits_row(b, r, got.data());
                score(s3, &ref[(size_t)(at + r) * V], got.data());
            }
            bolt_keep(b, keep);
            at += keep;
            cut = !cut;
        }
        report("verify windows 8/5", s3);
        return 0;
    }
    if (mode == "gen") {
        auto prompt = read_ids(argv[3]);
        int N = atoi(argv[4]);
        int draft = atoi(argv[5]);
        int reps = getenv("REPS") ? atoi(getenv("REPS")) : 1;
        for (int rep = 0; rep < reps; rep++) {
            bolt_reset(b);
            float v; int id;
            double t0 = now_ms();
            bolt_prefill(b, prompt.data(), (int)prompt.size(), 1);
            bolt_candidates(b, 0, 1, &v, &id);
            double t1 = now_ms();
            std::vector<int> out{id};
            int rounds = 0, accepted = 0;
            while ((int)out.size() < N) {
                int pending = out.back();
                if (draft) {
                    int d[8];
                    int got = bolt_propose(b, pending, d);
                    int win[8] = {pending};
                    for (int j = 0; j < 7; j++) win[1 + j] = got == 7 ? d[j] : 0;
                    const int w = got == 7 ? 8 : 1;
                    bolt_forward(b, win, w, 1, 1);
                    int n = 1;
                    std::vector<int> argmax(w);
                    for (int r = 0; r < w; r++) bolt_candidates(b, r, 1, &v, &argmax[r]);
                    while (n < w && argmax[n - 1] == win[n]) { out.push_back(win[n]); n++; }
                    bolt_keep(b, n);
                    out.push_back(argmax[n - 1]);
                    rounds++;
                    accepted += n - 1;
                } else {
                    bolt_forward(b, &pending, 1, 1, 1);
                    bolt_keep(b, 1);
                    bolt_candidates(b, 0, 1, &v, &id);
                    out.push_back(id);
                }
            }
            double t2 = now_ms();
            const int gen = (int)out.size() - 1;
            printf("prompt %zu tokens %.0f ms (%.0f tok/s); %d tokens in %.0f ms: %.1f tok/s", prompt.size(), t1 - t0, prompt.size() / ((t1 - t0) / 1000), gen, t2 - t1, gen / ((t2 - t1) / 1000));
            if (draft) printf("; %d rounds, %.2f drafts accepted a round, %.2f tokens a round", rounds, (double)accepted / rounds, (double)gen / rounds);
            printf("\n");
            if (rep == reps - 1 && getenv("SHOW")) {
                printf("ids:");
                for (int t : out) printf(" %d", t);
                printf("\n");
            }
            fflush(stdout);
        }
        bolt_prof_report(b);
        return 0;
    }
    if (mode == "steps") {   // steps MODEL IDS_FILE N [WIDTH]: N verify windows of WIDTH rows kept whole, after the prompt
        auto prompt = read_ids(argv[3]);
        const int N = atoi(argv[4]), W = argc > 5 ? atoi(argv[5]) : 1;
        bolt_prefill(b, prompt.data(), (int)prompt.size(), 1);
        bolt_prof_report(b);
        float v; int id;
        bolt_candidates(b, 0, 1, &v, &id);
        double t0 = now_ms();
        for (int i = 0; i < N; i++) {
            int win[8];
            for (int j = 0; j < W; j++) win[j] = id + j;
            bolt_forward(b, win, W, 1, 1);
            bolt_keep(b, W);
            bolt_candidates(b, W - 1, 1, &v, &id);
        }
        printf("%d windows of %d rows: %.2f ms each\n", N, W, (now_ms() - t0) / N);
        bolt_prof_report(b);
        return 0;
    }
    if (mode == "prefill") {
        int N = atoi(argv[3]);
        std::vector<int> ids(N);
        for (int i = 0; i < N; i++) ids[i] = 1000 + (int)((i * 2654435761u) % 60000u);
        for (int rep = 0; rep < 2; rep++) {
            bolt_reset(b);
            double t0 = now_ms();
            bolt_prefill(b, ids.data(), N, 1);
            double t1 = now_ms();
            printf("prefill %d tokens: %.0f ms, %.0f tok/s\n", N, t1 - t0, N / ((t1 - t0) / 1000));
            fflush(stdout);
        }
        return 0;
    }
    return 2;
}
