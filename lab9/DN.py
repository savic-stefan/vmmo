import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from time import sleep

from mdp import MDP, FrozenLakeEnv, has_graphviz, plot_graph, plot_graph_with_state_values, plot_graph_optimal_strategy_and_state_values

print("=" * 80)
print("ЗАДАНИЕ №4: МАРКОВСКИЙ ПРОЦЕСС ПРИНЯТИЯ РЕШЕНИЙ")
print("=" * 80)

print("\n1. ИНИЦИАЛИЗАЦИЯ MDP И ПРОВЕРКА РАБОТЫ")
print("-" * 60)

transition_probs = {
    's0': {
        'a0': {'s0': 0.5, 's2': 0.5},
        'a1': {'s2': 1}
    },
    's1': {
        'a0': {'s0': 0.7, 's1': 0.1, 's2': 0.2},
        'a1': {'s1': 0.95, 's2': 0.05}
    },
    's2': {
        'a0': {'s0': 0.4, 's2': 0.6},
        'a1': {'s0': 0.3, 's1': 0.3, 's2': 0.4}
    }
}

rewards = {
    's1': {'a0': {'s0': +5}},
    's2': {'a1': {'s0': -1}}
}

mdp = MDP(transition_probs, rewards, initial_state='s0')

print('Начальное состояние =', mdp.reset())
next_state, reward, done, info = mdp.step('a1')
print('Следующее состояние = %s, награда = %s, завершено = %s' % (next_state, reward, done))

print("\nМетоды MDP:")
print("Все состояния:", mdp.get_all_states())
print("Возможные действия в s1:", mdp.get_possible_actions('s1'))
print("Следующие состояния из s1 при действии a0:", mdp.get_next_states('s1', 'a0'))
print("Награда за переход из s1 в s0 при действии a0:", mdp.get_reward('s1', 'a0', 's0'))
print("Вероятность перехода из s1 в s0 при действии a0:", mdp.get_transition_prob('s1', 'a0', 's0'))

print("\nGraphviz доступен:", has_graphviz)

if has_graphviz:
    try:
        print("\nСохранение графа MDP в файл 'mdp_graph.png'...")
        graph = plot_graph(mdp)
        graph.format = 'png'
        graph.render('mdp_graph', view=False, cleanup=True)
        print("✓ Граф MDP сохранен как 'mdp_graph.png'")
    except Exception as e:
        print(f"✗ Ошибка при сохранении графа MDP: {e}")

print("\n2. ИТЕРАЦИИ ПО ЗНАЧЕНИЯМ (VALUE ITERATION)")
print("-" * 60)

def get_action_value(mdp, state_values, state, action, gamma):
    q = 0
    for next_state, prob in mdp.get_next_states(state, action).items():
        r = mdp.get_reward(state, action, next_state)
        q += prob * (r + gamma * state_values.get(next_state, 0))
    return q

def get_new_state_value(mdp, state_values, state, gamma):
    if mdp.is_terminal(state):
        return 0
    possible_actions = mdp.get_possible_actions(state)
    if not possible_actions:
        return 0
    q_values = [get_action_value(mdp, state_values, state, a, gamma) for a in possible_actions]
    return max(q_values)

print("Тестирование функций:")
test_Vs = {s: i for i, s in enumerate(sorted(mdp.get_all_states()))}
print(f"Q(s2, a1) = {get_action_value(mdp, test_Vs, 's2', 'a1', 0.9):.2f} (ожидается 0.69)")
print(f"Q(s1, a0) = {get_action_value(mdp, test_Vs, 's1', 'a0', 0.9):.2f} (ожидается 3.95)")

assert np.isclose(get_action_value(mdp, test_Vs, 's2', 'a1', 0.9), 0.69)
assert np.isclose(get_action_value(mdp, test_Vs, 's1', 'a0', 0.9), 3.95)
print("Тесты функций пройдены ✓")

print("\n3. ОСНОВНОЙ ЦИКЛ VALUE ITERATION")
print("-" * 60)

gamma = 0.9
num_iter = 100
min_difference = 0.001
state_values = {s: 0 for s in mdp.get_all_states()}

if has_graphviz:
    try:
        print("Сохранение графа с начальными значениями состояния...")
        graph_with_values = plot_graph_with_state_values(mdp, state_values)
        graph_with_values.format = 'png'
        graph_with_values.render('mdp_initial_values', view=False, cleanup=True)
        print("✓ Граф с начальными значениями сохранен как 'mdp_initial_values.png'")
    except Exception as e:
        print(f"✗ Ошибка: {e}")

print("\nПроцесс итераций:")
for i in range(num_iter):
    new_state_values = {}
    for s in mdp.get_all_states():
        new_state_values[s] = get_new_state_value(mdp, state_values, s, gamma)
    
    diff = max(abs(new_state_values[s] - state_values[s]) for s in mdp.get_all_states())
    print("iter %4i | diff: %.5f | " % (i, diff), end="")
    print(' '.join("V(%s) = %.3f" % (s, v) for s, v in state_values.items()))
    state_values = new_state_values
    if diff < min_difference:
        print("\nДостигнута требуемая точность!")
        break

print("\nФинальные значения состояния:")
for s, v in state_values.items():
    print(f"  V({s}) = {v:.6f}")

assert abs(state_values['s0'] - 3.781) < 0.01
assert abs(state_values['s1'] - 7.294) < 0.01
assert abs(state_values['s2'] - 4.202) < 0.01
print("Проверка значений состояния пройдена ✓")

if has_graphviz:
    try:
        print("\nСохранение графа с финальными значениями состояния...")
        graph_final = plot_graph_with_state_values(mdp, state_values)
        graph_final.format = 'png'
        graph_final.render('mdp_final_values', view=False, cleanup=True)
        print("✓ Граф с финальными значениями сохранен как 'mdp_final_values.png'")
    except Exception as e:
        print(f"✗ Ошибка: {e}")

print("\n4. ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОЙ ПОЛИТИКИ")
print("-" * 60)

def get_optimal_action(mdp, state_values, state, gamma=0.9):
    if mdp.is_terminal(state):
        return None
    possible_actions = mdp.get_possible_actions(state)
    if not possible_actions:
        return None
    q_values = {}
    for a in possible_actions:
        q_values[a] = get_action_value(mdp, state_values, state, a, gamma)
    return max(q_values, key=q_values.get)

print("Оптимальные действия для каждого состояния:")
print(f"  π*(s0) = {get_optimal_action(mdp, state_values, 's0', gamma)}")
print(f"  π*(s1) = {get_optimal_action(mdp, state_values, 's1', gamma)}")
print(f"  π*(s2) = {get_optimal_action(mdp, state_values, 's2', gamma)}")

assert get_optimal_action(mdp, state_values, 's0', gamma) == 'a1'
assert get_optimal_action(mdp, state_values, 's1', gamma) == 'a0'
assert get_optimal_action(mdp, state_values, 's2', gamma) == 'a1'
print("Проверка оптимальных действий пройдена ✓")

assert get_optimal_action(mdp, {'s0': -1e10, 's1': 0, 's2': -2e10}, 's0', 0.9) == 'a0'
assert get_optimal_action(mdp, {'s0': -2e10, 's1': 0, 's2': -1e10}, 's0', 0.9) == 'a1'
print("Тесты с отрицательными значениями Q пройдены ✓")

print("\n5. ОЦЕНКА КАЧЕСТВА ПОЛИТИКИ")
print("-" * 60)

s = mdp.reset()
rewards = []
for _ in range(10000):
    a = get_optimal_action(mdp, state_values, s, gamma)
    s, r, done, _ = mdp.step(a)
    rewards.append(r)

avg_reward = np.mean(rewards)
print(f"Средняя награда за 10000 шагов: {avg_reward:.4f}")
assert 0.40 < avg_reward < 0.55
print("Проверка средней награды пройдена ✓")

print("\n6. ЗАДАЧА FROZEN LAKE")
print("=" * 60)

print("\n6.1. Frozen Lake без скольжения (slip_chance=0)")
print("-" * 40)

mdp_fl = FrozenLakeEnv(slip_chance=0)
print("Карта Frozen Lake:")
mdp_fl.render()

def value_iteration(mdp, state_values=None, gamma=0.9, num_iter=1000, min_difference=1e-5):
    if state_values is None:
        state_values = {s: 0 for s in mdp.get_all_states()}
    
    for i in range(num_iter):
        new_state_values = {}
        for s in mdp.get_all_states():
            new_state_values[s] = get_new_state_value(mdp, state_values, s, gamma)
        
        diff = 0
        for s in mdp.get_all_states():
            change = abs(new_state_values[s] - state_values.get(s, 0))
            if change > diff:
                diff = change
        
        start_state = mdp._initial_state if hasattr(mdp, '_initial_state') else list(mdp.get_all_states())[0]
        start_value = new_state_values.get(start_state, 0)
        
        print("iter %4i | diff: %6.5f | V(start): %.3f " % (i, diff, start_value))
        state_values = new_state_values
        
        if diff < min_difference:
            break
    
    return state_values

state_values_fl = value_iteration(mdp_fl)

print("\nДемонстрация оптимальной траектории:")
s = mdp_fl.reset()
mdp_fl.render()
for t in range(100):
    a = get_optimal_action(mdp_fl, state_values_fl, s, gamma)
    print(a, end='\n\n')
    s, r, done, _ = mdp_fl.step(a)
    mdp_fl.render()
    if done:
        break

print("\n6.2. Визуализация эволюции политики (первые 10 итераций)")
print("-" * 60)

def draw_policy(mdp, state_values):
    plt.figure(figsize=(3, 3))
    h, w = mdp.desc.shape
    states = sorted(mdp.get_all_states())
    
    V = np.zeros((h, w))
    for s in states:
        if isinstance(s, tuple) and len(s) == 2:
            y, x = s
            V[y, x] = state_values.get(s, 0)
    
    Pi = {s: get_optimal_action(mdp, state_values, s, gamma) for s in states}
    plt.imshow(V.reshape(h, w), cmap='gray', interpolation='none', clim=(0, 1))
    
    ax = plt.gca()
    ax.set_xticks(np.arange(w)-.5)
    ax.set_yticks(np.arange(h)-.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    Y, X = np.mgrid[0:h, 0:w]
    a2uv = {'left': (-1, 0), 'down': (0, -1), 'right': (1, 0), 'up': (0, 1)}
    
    for y in range(h):
        for x in range(w):
            cell = mdp.desc[y, x]
            cell_str = ''
            try:
                if hasattr(cell, 'decode'):
                    cell_str = cell.decode()
                else:
                    cell_str = str(cell)
                    cell_str = cell_str.replace("b'", "").replace("'", "")
            except:
                cell_str = str(cell)
            
            plt.text(x, y, cell_str,
                    color='g', size=12, verticalalignment='center',
                    horizontalalignment='center', fontweight='bold')
            
            a = Pi.get((y, x))
            if a is None:
                continue
            u, v = a2uv[a]
            plt.arrow(x, y, u*.3, -v*.3, color='m',
                     head_width=0.1, head_length=0.1)
    
    plt.grid(color='b', lw=2, ls='-')
    plt.show()

state_values = {s: 0 for s in mdp_fl.get_all_states()}

for i in range(10):
    print(f"\nПосле итерации {i}:")
    state_values = value_iteration(mdp_fl, state_values, num_iter=1)
    draw_policy(mdp_fl, state_values)

print("\n6.3. Анимация для 8x8 Frozen Lake (slip_chance=0.1)")
print("-" * 60)

mdp_fl_8x8 = FrozenLakeEnv(map_name='8x8', slip_chance=0.1)
state_values_8x8 = {s: 0 for s in mdp_fl_8x8.get_all_states()}

for i in range(10):
    clear_output(wait=True)
    print(f"После итерации {i}")
    state_values_8x8 = value_iteration(mdp_fl_8x8, state_values_8x8, num_iter=1)
    draw_policy(mdp_fl_8x8, state_values_8x8)
    sleep(0.5)

print("\n6.4. Массовые испытания для разных параметров")
print("-" * 60)

print("\nТест 1: slip_chance = 0")
mdp_test1 = FrozenLakeEnv(slip_chance=0)
state_values_test1 = value_iteration(mdp_test1)

total_rewards = []
for game_i in range(1000):
    s = mdp_test1.reset()
    rewards = []
    for t in range(100):
        s, r, done, _ = mdp_test1.step(
            get_optimal_action(mdp_test1, state_values_test1, s, gamma))
        rewards.append(r)
        if done:
            break
    total_rewards.append(np.sum(rewards))

avg_reward = np.mean(total_rewards)
print(f"Средняя награда: {avg_reward:.4f}")
assert 1.0 <= avg_reward <= 1.0

print("\nТест 2: slip_chance = 0.1")
mdp_test2 = FrozenLakeEnv(slip_chance=0.1)
state_values_test2 = value_iteration(mdp_test2)

total_rewards = []
for game_i in range(1000):
    s = mdp_test2.reset()
    rewards = []
    for t in range(100):
        s, r, done, _ = mdp_test2.step(
            get_optimal_action(mdp_test2, state_values_test2, s, gamma))
        rewards.append(r)
        if done:
            break
    total_rewards.append(np.sum(rewards))

avg_reward = np.mean(total_rewards)
print(f"Средняя награда: {avg_reward:.4f}")
assert 0.8 <= avg_reward <= 0.95

print("\nТест 3: slip_chance = 0.25")
mdp_test3 = FrozenLakeEnv(slip_chance=0.25)
state_values_test3 = value_iteration(mdp_test3)

total_rewards = []
for game_i in range(1000):
    s = mdp_test3.reset()
    rewards = []
    for t in range(100):
        s, r, done, _ = mdp_test3.step(
            get_optimal_action(mdp_test3, state_values_test3, s, gamma))
        rewards.append(r)
        if done:
            break
    total_rewards.append(np.sum(rewards))

avg_reward = np.mean(total_rewards)
print(f"Средняя награда: {avg_reward:.4f}")
assert 0.6 <= avg_reward <= 0.7

print("\nТест 4: slip_chance = 0.2, карта 8x8")
mdp_test4 = FrozenLakeEnv(slip_chance=0.2, map_name='8x8')
state_values_test4 = value_iteration(mdp_test4)

total_rewards = []
for game_i in range(1000):
    s = mdp_test4.reset()
    rewards = []
    for t in range(100):
        s, r, done, _ = mdp_test4.step(
            get_optimal_action(mdp_test4, state_values_test4, s, gamma))
        rewards.append(r)
        if done:
            break
    total_rewards.append(np.sum(rewards))

avg_reward = np.mean(total_rewards)
print(f"Средняя награда: {avg_reward:.4f}")
assert 0.6 <= avg_reward <= 0.8

print("\n" + "=" * 80)
print("РЕЗУЛЬТАТЫ:")
print("1. Графы MDP (Graphviz) сохранены в файлы (не отображаются в терминале):")
print("   - mdp_graph.png (оригинальный граф)")
print("   - mdp_initial_values.png (начальные значения)")
print("   - mdp_final_values.png (финальные значения)")
print("\n2. Графики Frozen Lake (matplotlib) отображены в отдельных окнах")
print("\n3. Все тесты выполнены успешно!")
print("=" * 80)