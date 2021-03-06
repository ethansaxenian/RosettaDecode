/**
 * @brief
 * [A* search algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
 * @details
 * A* is an informed search algorithm, or a best-first search, meaning that it
 * is formulated in terms of weighted graphs: starting from a specific starting
 * node of a graph (initial state), it aims to find a path to the given goal
 * node having the smallest cost (least distance travelled, shortest time,
 * etc.). It evaluates by maintaining a tree of paths originating at the start
 * node and extending those paths one edge at a time until it reaches the final
 * state.
 * The weighted edges (or cost) is evaluated on two factors, G score
 * (cost required from starting node or initial state to current state) and H
 * score (cost required from current state to final state). The F(state), then
 * is evaluated as:
 * F(state) = G(state) + H(state).
 *
 * To solve the given search with shortest cost or path possible  is to inspect
 * values having minimum F(state).
 * @author [Ashish Daulatabad](https://github.com/AshishYUO)
 */
#include <algorithm>   /// for `std::reverse` function
#include <array>       /// for `std::array`, representing `EightPuzzle` board
#include <cassert>     /// for `assert`
#include <functional>  /// for `std::function` STL
#include <iostream>    /// for IO operations
#include <map>         /// for `std::map` STL
#include <set>         /// for `std::set` STL
#include <vector>      /// for `std::vector` STL
/**
 * @namespace machine_learning
 * @brief Machine learning algorithms
 */
namespace machine_learning {
/**
 * @namespace aystar_search
 * @brief Functions for [A*
 * Search](https://en.wikipedia.org/wiki/A*_search_algorithm) implementation.
 */
namespace aystar_search {
/**
 * @class EightPuzzle
 * @brief A class defining [EightPuzzle/15-Puzzle
 * game](https://en.wikipedia.org/wiki/15_puzzle).
 * @details
 * A well known 3 x 3 puzzle of the form
 * `
 * 1   2   3
 * 4   5   6
 * 7   8   0
 * `
 * where `0` represents an empty space in the puzzle
 * Given any random state, the goal is to achieve the above configuration
 * (or any other configuration if possible)
 * @tparam N size of the square Puzzle, default is set to 3 (since it is
 * EightPuzzle)
 */
template <size_t N = 3>
class EightPuzzle {
    std::array<std::array<uint32_t, N>, N>
        board;  /// N x N array to store the current state of the Puzzle.

    std::vector<std::pair<int, int>> moves = {
        {0, 1},
        {1, 0},
        {0, -1},
        {-1,
         0}};  /// A helper array to evaluate the next state from current state;
    /**
     * @brief Finds an empty space in puzzle (in this case; a zero)
     * @returns a pair indicating integer distances from top and right
     * respectively, else returns -1, -1
     */
    std::pair<uint32_t, uint32_t> find_zero() {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (!board[i][j]) {
                    return {i, j};
                }
            }
        }
        return {-1, -1};
    }
    /**
     * @brief check whether the index value is bounded within the puzzle area
     * @param value index for the current board
     * @returns `true` if index is within the board, else `false`
     */
    inline bool in_range(const uint32_t value) const {
        return value >= 0 && value < N;
    }

 public:
    /**
     * @brief get the value from i units from right and j units from left side
     * of the board
     * @param i integer denoting ith row
     * @param j integer denoting column
     * @returns non-negative integer denoting the value at ith row and jth
     * column
     * @returns -1 if invalid i or j position
     */
    uint32_t get(size_t i, size_t j) const {
        if (in_range(i) && in_range(j)) {
            return board[i][j];
        }
        return -1;
    }
    /**
     * @brief Returns the current state of the board
     */
    std::array<std::array<uint32_t, N>, N> get_state() { return board; }

    /**
     * @brief returns the size of the EightPuzzle (number of row / column)
     * @return N, the size of the puzzle.
     */
    inline size_t get_size() const { return N; }
    /**
     * @brief Default constructor for EightPuzzle
     */
    EightPuzzle() {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                board[i][j] = ((i * 3 + j + 1) % (N * N));
            }
        }
    }
    /**
     * @brief Parameterized Constructor for EightPuzzle
     * @param init a 2-dimensional array denoting a puzzle configuration
     */
    explicit EightPuzzle(const std::array<std::array<uint32_t, N>, N> &init)
        : board(init) {}

    /**
     * @brief Copy constructor
     * @param A a reference of an EightPuzzle
     */
    EightPuzzle(const EightPuzzle<N> &A) : board(A.board) {}

    /**
     * @brief Move constructor
     * @param A a reference of an EightPuzzle
     */
    EightPuzzle(const EightPuzzle<N> &&A) noexcept
        : board(std::move(A.board)) {}
    /**
     * @brief Destructor of EightPuzzle
     */
    ~EightPuzzle() = default;

    /**
     * @brief Copy assignment operator
     * @param A a reference of an EightPuzzle
     */
    EightPuzzle &operator=(const EightPuzzle &A) {
        board = A.board;
        return *this;
    }

    /**
     * @brief Move assignment operator
     * @param A a reference of an EightPuzzle
     */
    EightPuzzle &operator=(EightPuzzle &&A) noexcept {
        board = std::move(A.board);
        return *this;
    }

    /**
     * @brief Find all possible states after processing all possible
     * moves, given the current state of the puzzle
     * @returns list of vector containing all possible next moves
     * @note the implementation is compulsory to create A* search
     */
    std::vector<EightPuzzle<N>> generate_possible_moves() {
        auto zero_pos = find_zero();
        // vector which will contain all possible state from current state
        std::vector<EightPuzzle<N>> NewStates;
        for (auto &move : moves) {
            if (in_range(zero_pos.first + move.first) &&
                in_range(zero_pos.second + move.second)) {
                // swap with the possible moves
                std::array<std::array<uint32_t, N>, N> new_config = board;
                std::swap(new_config[zero_pos.first][zero_pos.second],
                          new_config[zero_pos.first + move.first]
                                    [zero_pos.second + move.second]);
                EightPuzzle<N> new_state(new_config);
                // Store new state and calculate heuristic value, and depth
                NewStates.emplace_back(new_state);
            }
        }
        return NewStates;
    }
    /**
     * @brief check whether two boards are equal
     * @returns `true` if check.state is equal to `this->state`, else
     * `false`
     */
    bool operator==(const EightPuzzle<N> &check) const {
        if (check.get_size() != N) {
            return false;
        }
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (board[i][j] != check.board[i][j]) {
                    return false;
                }
            }
        }
        return true;
    }
    /**
     * @brief check whether one board is lexicographically smaller
     * @returns `true` if this->state is lexicographically smaller than
     * `check.state`, else `false`
     */
    bool operator<(const EightPuzzle<N> &check) const {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (board[i][j] != check.board[i][j]) {
                    return board[i][j] < check.board[i][j];
                }
            }
        }
        return false;
    }
    /**
     * @brief check whether one board is lexicographically smaller or equal
     * @returns `true` if this->state is lexicographically smaller than
     * `check.state` or same, else `false`
     */
    bool operator<=(const EightPuzzle<N> &check) const {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (board[i][j] != check.board[i][j]) {
                    return board[i][j] < check.board[i][j];
                }
            }
        }
        return true;
    }

    /**
     * @brief friend operator to display EightPuzzle<>
     * @param op ostream object
     * @param SomeState a certain state.
     * @returns ostream operator op
     */
    friend std::ostream &operator<<(std::ostream &op,
                                    const EightPuzzle<N> &SomeState) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                op << SomeState.board[i][j] << " ";
            }
            op << "\n";
        }
        return op;
    }
};
/**
 * @class AyStarSearch
 * @brief A class defining [A* search
 * algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm). for some
 * initial state and final state
 * @details AyStarSearch class is defined as the informed search algorithm
 * that is formulated in terms of weighted graphs: starting from a specific
 * starting node of a graph (initial state), it aims to find a path to the given
 * goal node having the smallest cost (least distance travelled, shortest time,
 * etc.)
 * The weighted edges (or cost) is evaluated on two factors, G score
 * (cost required from starting node or initial state to current state) and H
 * score (cost required from current state to final state). The `F(state)`, then
 * is evaluated as:
 * `F(state) = G(state) + H(state)`.
 * The best search would be the final state having minimum `F(state)` value
 * @tparam Puzzle denotes the puzzle or problem involving initial state and
 * final state to be solved by A* search.
 * @note 1. The algorithm is referred from pesudocode from
 * [Wikipedia page](https://en.wikipedia.org/wiki/A*_search_algorithm)
 * as is.
 * 2. For `AyStarSearch` to work, the definitions for template Puzzle is
 * compulsory.
 * a. Comparison operator for template Puzzle (`<`, `==`, and `<=`)
 * b. `generate_possible_moves()`
 */
template <typename Puzzle>
class AyStarSearch {
    /**
     * @brief Struct that handles all the information related to the current
     * state.
     */
    typedef struct Info {
        Puzzle state;                /// Holds the current state.
        size_t heuristic_value = 0;  /// stores h score
        size_t depth = 0;            /// stores g score

        /**
         * @brief Default constructor
         */
        Info() = default;

        /**
         * @brief constructor having Puzzle as parameter
         * @param A a puzzle object
         */
        explicit Info(Puzzle A) : state(std::move(A)) {}

        /**
         * @brief constructor having three parameters
         * @param A a puzzle object
         * @param h_value heuristic value of this puzzle object
         * @param depth the depth at which this node was found during traversal
         */
        Info(Puzzle A, size_t h_value, size_t d)
            : state(std::move(A)), heuristic_value(h_value), depth(d) {}

        /**
         * @brief Copy constructor
         * @param A Info object reference
         */
        Info(const Info &A)
            : state(A.state),
              heuristic_value(A.heuristic_value),
              depth(A.depth) {}

        /**
         * @brief Move constructor
         * @param A Info object reference
         */
        Info(const Info &&A) noexcept
            : state(std::move(A.state)),
              heuristic_value(std::move(A.heuristic_value)),
              depth(std::move(A.depth)) {}

        /**
         * @brief copy assignment operator
         * @param A Info object reference
         */
        Info &operator=(const Info &A) {
            state = A.state;
            heuristic_value = A.heuristic_value;
            depth = A.depth;
            return *this;
        }

        /**
         * @brief move assignment operator
         * @param A Info object reference
         */
        Info &operator=(Info &&A) noexcept {
            state = std::move(A.state);
            heuristic_value = std::move(A.heuristic_value);
            depth = std::move(A.depth);
            return *this;
        }
        /**
         * @brief Destructor for Info
         */
        ~Info() = default;
    } Info;

    Info Initial;  // Initial state of the AyStarSearch
    Info Final;    // Final state of the AyStarSearch
    /**
     * @brief Custom comparator for open_list
     */
    struct comparison_operator {
        bool operator()(const Info &a, const Info &b) const {
            return a.state < b.state;
        }
    };

 public:
    /**
     * @brief Parameterized constructor for AyStarSearch
     * @param initial denoting initial state of the puzzle
     * @param final denoting final state of the puzzle
     */
    AyStarSearch(const Puzzle &initial, const Puzzle &final) {
        Initial = Info(initial);
        Final = Info(final);
    }
    /**
     * @brief A helper solution: launches when a solution for AyStarSearch
     * is found
     * @param FinalState the pointer to the obtained final state
     * @param parent_of the list of all parents of nodes stored during A*
     * search
     * @returns the list of moves denoting moves from final state to initial
     * state (in reverse)
     */
    std::vector<Puzzle> Solution(
        Info *FinalState,
        const std::map<Info, Info *, comparison_operator> &parent_of) {
        //  Useful for traversing from final state to current state.
        Info *current_state = FinalState;
        /*
         * For storing the solution tree starting from initial state to
         * final state
         */
        std::vector<Puzzle> answer;
        while (current_state != nullptr) {
            answer.emplace_back(current_state->state);
            current_state = parent_of.find(*current_state)->second;
        }
        return answer;
    }
    /**
     * Main algorithm for finding `FinalState`, given the `InitialState`
     * @param dist the heuristic finction, defined by the user
     * @param permissible_depth the depth at which the A* search discards
     * searching for solution
     * @returns List of moves from Final state to initial state, if
     * evaluated, else returns an empty array
     */
    std::vector<Puzzle> a_star_search(
        const std::function<uint32_t(const Puzzle &, const Puzzle &)> &dist,
        const uint32_t permissible_depth = 30) {
        std::map<Info, Info *, comparison_operator>
            parent_of;  /// Stores the parent of the states
        std::map<Info, uint32_t, comparison_operator>
            g_score;  /// Stores the g_score
        std::set<Info, comparison_operator>
            open_list;  /// Stores the list to explore
        std::set<Info, comparison_operator>
            closed_list;  /// Stores the list that are explored

        // Before starting the AyStartSearch, initialize the set and maps
        open_list.emplace(Initial);
        parent_of[Initial] = nullptr;
        g_score[Initial] = 0;

        while (!open_list.empty()) {
            // Iterator for state having having lowest f_score.
            typename std::set<Info, comparison_operator>::iterator
                it_low_f_score;
            uint32_t min_f_score = 1e9;
            for (auto iter = open_list.begin(); iter != open_list.end();
                 ++iter) {
                // f score here is evaluated by g score (depth) and h score
                // (distance between current state and final state)
                uint32_t f_score = iter->heuristic_value + iter->depth;
                if (f_score < min_f_score) {
                    min_f_score = f_score;
                    it_low_f_score = iter;
                }
            }

            // current_state, stores lowest f score so far for this state.
            Info *current_state = new Info(*it_low_f_score);

            // if this current state is equal to final, return
            if (current_state->state == Final.state) {
                return Solution(current_state, parent_of);
            }
            // else remove from open list as visited.
            open_list.erase(it_low_f_score);
            // if current_state has exceeded the allowed depth, skip
            // neighbor checking
            if (current_state->depth >= permissible_depth) {
                continue;
            }
            // Generate all possible moves (neighbors) given the current
            // state
            std::vector<Puzzle> total_possible_moves =
                current_state->state.generate_possible_moves();

            for (Puzzle &neighbor : total_possible_moves) {
                // calculate score of neighbors with respect to
                // current_state
                Info Neighbor = {neighbor, dist(neighbor, Final.state),
                                 current_state->depth + 1};
                uint32_t temp_g_score = Neighbor.depth;

                // Check whether this state is explored.
                // If this state is discovered at greater depth, then discard,
                // else remove from closed list and explore the node
                auto closed_list_iter = closed_list.find(Neighbor);
                if (closed_list_iter != closed_list.end()) {
                    // 1. If state in closed list has higher depth, then remove
                    // from list since we have found better option,
                    // 2. Else don't explore this state.
                    if (Neighbor.depth < closed_list_iter->depth) {
                        closed_list.erase(closed_list_iter);
                    } else {
                        continue;
                    }
                }
                auto neighbor_g_score_iter = g_score.find(Neighbor);
                // if the neighbor is already created and has minimum
                // g_score, then update g_score and f_score else insert new
                if (neighbor_g_score_iter != g_score.end()) {
                    if (neighbor_g_score_iter->second > temp_g_score) {
                        neighbor_g_score_iter->second = temp_g_score;
                        parent_of[Neighbor] = current_state;
                    }
                } else {
                    g_score[Neighbor] = temp_g_score;
                    parent_of[Neighbor] = current_state;
                }
                // If this is a new state, insert into open_list
                // else update if the this state has better g score than
                // existing one.
                auto iter = open_list.find(Neighbor);
                if (iter == open_list.end()) {
                    open_list.emplace(Neighbor);
                } else if (iter->depth > Neighbor.depth) {
                    open_list.erase(iter);
                    open_list.emplace(Neighbor);
                }
            }
            closed_list.emplace(*current_state);
        }
        // Cannot find the solution, return empty vector
        return std::vector<Puzzle>(0);
    }
};
}  // namespace aystar_search
}  // namespace machine_learning

/**
 * @brief Self test-implementations
 * @returns void
 */
static void test() {
    // Renaming for simplicity
    using matrix3 = std::array<std::array<uint32_t, 3>, 3>;
    using row3 = std::array<uint32_t, 3>;
    using matrix4 = std::array<std::array<uint32_t, 4>, 4>;
    using row4 = std::array<uint32_t, 4>;
    // 1st test: A* search for simple EightPuzzle problem
    matrix3 puzzle;
    puzzle[0] = row3({0, 2, 3});
    puzzle[1] = row3({1, 5, 6});
    puzzle[2] = row3({4, 7, 8});

    matrix3 ideal;
    ideal[0] = row3({1, 2, 3});
    ideal[1] = row3({4, 5, 6});
    ideal[2] = row3({7, 8, 0});

    /*
     * Heuristic function: Manhattan distance
     */
    auto manhattan_distance =
        [](const machine_learning::aystar_search::EightPuzzle<> &first,
           const machine_learning::aystar_search::EightPuzzle<> &second) {
            uint32_t ret = 0;
            for (int i = 0; i < first.get_size(); ++i) {
                for (int j = 0; j < first.get_size(); ++j) {
                    uint32_t find = first.get(i, j);
                    int m = -1, n = -1;
                    for (int k = 0; k < second.get_size(); ++k) {
                        for (int l = 0; l < second.get_size(); ++l) {
                            if (find == second.get(k, l)) {
                                std::tie(m, n) = std::make_pair(k, l);
                                break;
                            }
                        }
                        if (m != -1) {
                            break;
                        }
                    }
                    if (m != -1) {
                        ret += abs(m - i) + abs(n - j);
                    }
                }
            }
            return ret;
        };

    machine_learning::aystar_search::EightPuzzle<> Puzzle(puzzle);
    machine_learning::aystar_search::EightPuzzle<> Ideal(ideal);
    machine_learning::aystar_search::AyStarSearch<
        machine_learning::aystar_search::EightPuzzle<3>>
        search(Puzzle, Ideal);  /// Search object

    std::vector<matrix3> answer;  /// Array that validates the answer

    answer.push_back(
        matrix3({row3({0, 2, 3}), row3({1, 5, 6}), row3({4, 7, 8})}));
    answer.push_back(
        matrix3({row3({1, 2, 3}), row3({0, 5, 6}), row3({4, 7, 8})}));
    answer.push_back(
        matrix3({row3({1, 2, 3}), row3({4, 5, 6}), row3({0, 7, 8})}));
    answer.push_back(
        matrix3({row3({1, 2, 3}), row3({4, 5, 6}), row3({7, 0, 8})}));
    answer.push_back(
        matrix3({row3({1, 2, 3}), row3({4, 5, 6}), row3({7, 8, 0})}));

    auto Solution = search.a_star_search(manhattan_distance);
    std::cout << Solution.size() << std::endl;

    assert(Solution.size() == answer.size());

    uint32_t i = 0;
    for (auto it = Solution.rbegin(); it != Solution.rend(); ++it) {
        assert(it->get_state() == answer[i]);
        ++i;
    }

    // 2nd test: A* search for complicated EightPuzzle problem
    // Initial state
    puzzle[0] = row3({5, 7, 3});
    puzzle[1] = row3({2, 0, 6});
    puzzle[2] = row3({1, 4, 8});
    // Final state
    ideal[0] = row3({1, 2, 3});
    ideal[1] = row3({4, 5, 6});
    ideal[2] = row3({7, 8, 0});

    Puzzle = machine_learning::aystar_search::EightPuzzle<>(puzzle);
    Ideal = machine_learning::aystar_search::EightPuzzle<>(ideal);

    // Initialize the search object
    search = machine_learning::aystar_search::AyStarSearch<
        machine_learning::aystar_search::EightPuzzle<3>>(Puzzle, Ideal);

    Solution = search.a_star_search(manhattan_distance);
    std::cout << Solution.size() << std::endl;
    // Static assertion due to large solution
    assert(13 == Solution.size());
    // Check whether the final state is equal to expected one
    assert(Solution[0].get_state() == ideal);
    for (auto it = Solution.rbegin(); it != Solution.rend(); ++it) {
        std::cout << *it << std::endl;
    }

    // 3rd test: A* search for 15-Puzzle
    // Initial State of the puzzle
    matrix4 puzzle2;
    puzzle2[0] = row4({5, 1, 2, 3});
    puzzle2[1] = row4({9, 6, 8, 4});
    puzzle2[2] = row4({13, 10, 7, 11});
    puzzle2[3] = row4({14, 15, 12, 0});
    // Final state of the puzzle
    matrix4 ideal2;
    ideal2[0] = row4({1, 2, 3, 4});
    ideal2[1] = row4({5, 6, 7, 8});
    ideal2[2] = row4({9, 10, 11, 12});
    ideal2[3] = row4({13, 14, 15, 0});

    // Instantiate states for a*, initial state and final states
    machine_learning::aystar_search::EightPuzzle<4> Puzzle2(puzzle2),
        Ideal2(ideal2);
    // Initialize the search object
    machine_learning::aystar_search::AyStarSearch<
        machine_learning::aystar_search::EightPuzzle<4>>
        search2(Puzzle2, Ideal2);
    /**
     * Heuristic function: Manhattan distance
     */
    auto manhattan_distance2 =
        [](const machine_learning::aystar_search::EightPuzzle<4> &first,
           const machine_learning::aystar_search::EightPuzzle<4> &second) {
            uint32_t ret = 0;
            for (int i = 0; i < first.get_size(); ++i) {
                for (int j = 0; j < first.get_size(); ++j) {
                    uint32_t find = first.get(i, j);
                    int m = -1, n = -1;
                    for (int k = 0; k < second.get_size(); ++k) {
                        for (int l = 0; l < second.get_size(); ++l) {
                            if (find == second.get(k, l)) {
                                std::tie(m, n) = std::make_pair(k, l);
                                break;
                            }
                        }
                        if (m != -1) {
                            break;
                        }
                    }
                    if (m != -1) {
                        ret += abs(m - i) + abs(n - j);
                    }
                }
            }
            return ret;
        };

    auto sol2 = search2.a_star_search(manhattan_distance2);
    std::cout << sol2.size() << std::endl;

    // Static assertion due to large solution
    assert(15 == sol2.size());
    // Check whether the final state is equal to expected one
    assert(sol2[0].get_state() == ideal2);

    for (auto it = sol2.rbegin(); it != sol2.rend(); ++it) {
        std::cout << *it << std::endl;
    }
}
/**
 * @brief Main function
 * @returns 0 on exit
 */
int main() {
    test();  // run self-test implementations
    return 0;
}
