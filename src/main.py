# main.py
from environment import Environment
from agent import QLearning, SARSA

def play_game():
    env = Environment()
    print("Pilih algoritma untuk bermain: ")
    print("1. Q-Learning")
    print("2. SARSA")
    print("3. Keduanya")

    choice = int(input("Enter your choice: "))

    if choice == 1:
        q_learning = QLearning(env)
        score, path, win_episode, lose_episode = q_learning.train()
        print("\nQ-Learning completed.")
    elif choice == 2:
        sarsa = SARSA(env)
        score, path, win_episode, lose_episode = sarsa.train()
        print("\nSARSA completed.")
    elif choice == 3:
        q_learning = QLearning(env)
        q_score, q_path, q_win_episode, q_lose_episode = q_learning.train()
        print("\nQ-Learning completed.")
        sarsa = SARSA(env)
        s_score, s_path, s_win_episode, s_lose_episode = sarsa.train()
        print("SARSA completed.")
    else:
        print("\nInvalid choice!")
        return


    if choice != 3:
        print(f"\nFinal Score: {score}")
        if win_episode != -1:
            print(f"Model won at episode: {win_episode}")
        if lose_episode != -1:
            print(f"Model lost at episode: {lose_episode}")

        if choice == 1:
            q_learning.q_table.print_q_table()
        else:
            sarsa.q_table.print_q_table()

        print("\nPath taken:", " -> ".join(map(str, path)))

    else:
        print(f"\nQ-Learning final Score: {q_score}")
        print(f"SARSA final Score: {s_score}")
        if q_win_episode != -1:
            print(f"Q-Learning won at episode: {q_win_episode}")
        if s_win_episode != -1:
            print(f"SARSA won at episode: {s_win_episode}")
        if q_lose_episode != -1:
            print(f"Q-Learning lost at episode: {q_lose_episode}")
        if s_lose_episode != -1:
            print(f"SARSA lost at episode: {s_lose_episode}")
        
        print("\nQ-Learning Q-Table:")
        q_learning.q_table.print_q_table()
        print("\nPath taken:", " -> ".join(map(str, q_path)))
        
        print("\nSARSA Q-Table:")
        q_learning.q_table.print_q_table()
        print("\nPath taken:", " -> ".join(map(str, s_path)))

if __name__ == "__main__":
    play_game()
