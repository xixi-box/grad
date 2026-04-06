import { motion } from 'framer-motion'

const technologies = [
  { name: 'PyTorch', description: 'Deep Learning Framework', color: '#EE4C2C' },
  { name: 'CUDA', description: 'GPU Computing', color: '#76B900' },
  { name: 'gsplat', description: '3D Gaussian Splatting', color: '#8B5CF6' },
  { name: 'Open3D', description: '3D Visualization', color: '#3B82F6' },
  { name: 'DUSt3R', description: 'Depth Estimation', color: '#F59E0B' },
]

export default function TechStack() {
  return (
    <section className="py-16 relative bg-slate-900/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-wrap justify-center gap-4">
          {technologies.map((tech, index) => (
            <motion.div
              key={tech.name}
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
              whileHover={{ y: -4, scale: 1.05 }}
              className="flex items-center gap-3 px-5 py-3 rounded-xl bg-slate-800/50 border border-slate-700 hover:border-slate-600 transition-all"
            >
              <div
                className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-xs"
                style={{ backgroundColor: tech.color }}
              >
                {tech.name.charAt(0)}
              </div>
              <div>
                <div className="font-medium text-sm">{tech.name}</div>
                <div className="text-xs text-slate-500">{tech.description}</div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}